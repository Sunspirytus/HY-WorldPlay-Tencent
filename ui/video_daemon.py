#!/usr/bin/env python3
"""
HY-WorldPlay 视频生成守护进程

功能：
1. 轮询检查是否有待生成的视频任务（status=pending）
2. 检查GPU是否被其他程序占用
3. 使用多进程文件锁确保安全
4. 参考 run-1.sh 脚本执行视频生成任务
5. 更新任务状态到 metadata.json

使用方法:
    python video_daemon.py           # 前台运行
    python video_daemon.py --daemon  # 后台守护进程模式
    python video_daemon.py --stop    # 停止守护进程
"""

import os
import sys
import time
import json
import fcntl
import signal
import argparse
import subprocess
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

# 获取项目根目录（ui的父目录）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 添加项目路径（当前在ui文件夹内，需要添加父目录）
sys.path.insert(0, PROJECT_ROOT)

from ui.config import DATA_DIR, VIDEO_DIR, METADATA_FILE

from ui.metadata_manager import metadata_manager
from ui.gpu_monitor import get_gpu_info, init_gpu_monitor

# ==================== 配置 ====================

DAEMON_PID_FILE = os.path.join(PROJECT_ROOT, DATA_DIR, "video_daemon.pid")
DAEMON_LOCK_FILE = os.path.join(PROJECT_ROOT, DATA_DIR, "video_daemon.lock")
DAEMON_LOG_FILE = os.path.join(PROJECT_ROOT, DATA_DIR, "video_daemon.log")

# GPU 检查阈值 (MB) - 如果显存占用超过此值，认为GPU被占用
GPU_MEMORY_THRESHOLD_MB = 500

# 轮询间隔 (秒)
POLL_INTERVAL = 10

# 模型路径配置
MODEL_PATH = "./models/tencent---HunyuanVideo-1.5"
AR_DISTILL_ACTION_MODEL_PATH = "./models/tencent---HY-WorldPlay/ar_distilled_action_model/diffusion_pytorch_model.safetensors"

# 视频生成配置
RESOLUTION = "480p"
ASPECT_RATIO = "16:9"
SEED = 1


def get_available_gpus() -> tuple[int, str] | None:
    """
    检测可用的GPU数量和设备ID
    
    策略（优先使用少量GPU）：
    - 1卡: 显存占用<500M 且 总显存>70G
    - 2卡: 每张显存占用<500M 且 每张显存>50G
    - 4卡: 每张显存占用<500M 且 每张显存>30G
    - 8卡: 每张显存占用<500M 且 每张显存>23G
    
    Returns:
        (可用GPU数量, CUDA_VISIBLE_DEVICES字符串) 或 None（无可用GPU）
    """
    try:
        if not init_gpu_monitor():
            logger.warning("GPU监控初始化失败")
            return None
        
        gpu_info = get_gpu_info()
        if not gpu_info:
            logger.warning("无法获取GPU信息")
            return None
        
        # 收集所有空闲GPU的信息
        idle_gpus = []  # [(gpu_id, mem_total_gb), ...]
        for gpu in gpu_info:
            gpu_id = gpu['id']
            mem_total = float(gpu.get('mem_total', 0))  # GB
            processes = gpu.get('processes', [])
            
            # 计算当前显存占用
            mem_used_mb = sum(proc.get('mem_used_mb', 0) for proc in processes)
            
            # 检查是否被外部用户占用（显存占用超过阈值）
            is_occupied = False
            for proc in processes:
                proc_mem_mb = proc.get('mem_used_mb', 0)
                user = proc.get('user', 'unknown')
                if proc_mem_mb > GPU_MEMORY_THRESHOLD_MB:
                    if 'hy-worldplay' not in user.lower() and 'hunyuan' not in user.lower():
                        is_occupied = True
                        logger.debug(f"GPU {gpu_id} 被用户 {user} 占用 ({proc_mem_mb:.0f}MB)，跳过")
                        break
            
            if not is_occupied and mem_used_mb <= GPU_MEMORY_THRESHOLD_MB:
                idle_gpus.append((gpu_id, mem_total, mem_used_mb))
                logger.debug(f"GPU {gpu_id} 空闲: 总显存={mem_total:.1f}GB, 已用={mem_used_mb:.0f}MB")
        
        if not idle_gpus:
            logger.warning("没有可用的GPU，等待下一轮轮询")
            return None
        
        # 按显存大小排序（优先使用大显存的卡）
        idle_gpus.sort(key=lambda x: x[1], reverse=True)
        
        # 定义策略：(GPU数量, 单卡最小显存GB)
        strategies = [
            (1, 70),  # 1卡需要>70G
            (2, 50),  # 2卡每张需要>50G
            (4, 30),  # 4卡每张需要>30G
            (8, 23),  # 8卡每张需要>23G
        ]
        
        # 尝试每种策略（优先使用少量GPU）
        for num_gpus, min_mem_gb in strategies:
            if len(idle_gpus) >= num_gpus:
                # 检查前num_gpus张卡是否都满足显存要求
                selected = idle_gpus[:num_gpus]
                all_meet_requirement = all(mem >= min_mem_gb for _, mem, _ in selected)
                
                if all_meet_requirement:
                    gpu_ids = [str(gpu_id) for gpu_id, _, _ in selected]
                    cuda_devices = ",".join(gpu_ids)
                    logger.info(f"策略匹配: 使用 {num_gpus} 张GPU ({cuda_devices}), 每张显存>{min_mem_gb}G")
                    return num_gpus, cuda_devices
        
        # 如果没有策略匹配，返回None（不强制使用任何GPU）
        logger.warning(f"没有满足策略的GPU组合，空闲GPU: {len(idle_gpus)}张，等待下一轮轮询")
        return None
    
    except Exception as e:
        logger.error(f"检测GPU时出错: {e}")
        return None


# ==================== 日志工具 ====================
class Logger:
    """简单的日志记录器（启动时清空，支持大文件自动清空）"""
    
    MAX_LOG_SIZE = 10 * 1024 * 1024  # 10MB
    
    def __init__(self, log_file: str, clear_on_start: bool = True):
        self.log_file = log_file
        self._lock = threading.Lock()
        # 确保日志目录存在
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        # 启动时清空日志
        if clear_on_start:
            self._clear_log()
    
    def _clear_log(self):
        """清空日志文件"""
        try:
            with open(self.log_file, 'w', encoding='utf-8') as f:
                f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [INFO] 日志已清空，守护进程启动\n")
        except Exception as e:
            print(f"清空日志失败: {e}")
    
    def _check_and_rotate(self):
        """检查日志文件大小，超过阈值则清空"""
        try:
            if os.path.exists(self.log_file):
                file_size = os.path.getsize(self.log_file)
                if file_size > self.MAX_LOG_SIZE:
                    # 清空日志文件
                    with open(self.log_file, 'w', encoding='utf-8') as f:
                        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [INFO] 日志文件超过10MB，已自动清空\n")
        except Exception as e:
            print(f"日志轮转失败: {e}")
    
    def _write(self, level: str, message: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] [{level}] {message}\n"
        
        with self._lock:
            # 检查并执行日志轮转
            self._check_and_rotate()
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(log_line)
        
        # 同时输出到控制台
        print(log_line.strip())
    
    def info(self, message: str):
        self._write("INFO", message)
    
    def warning(self, message: str):
        self._write("WARNING", message)
    
    def error(self, message: str):
        self._write("ERROR", message)
    
    def debug(self, message: str):
        self._write("DEBUG", message)


logger = Logger(DAEMON_LOG_FILE)


# ==================== 文件锁 ====================
class FileLock:
    """
    多进程文件锁
    用于确保只有一个守护进程实例在运行
    """
    
    def __init__(self, lock_file: str):
        self.lock_file = lock_file
        self.fd = None
    
    def acquire(self, blocking: bool = True) -> bool:
        """
        获取文件锁
        
        Args:
            blocking: 是否阻塞等待
            
        Returns:
            是否成功获取锁
        """
        try:
            self.fd = open(self.lock_file, 'w')
            if blocking:
                fcntl.flock(self.fd.fileno(), fcntl.LOCK_EX)
            else:
                fcntl.flock(self.fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            # 写入PID
            self.fd.write(str(os.getpid()))
            self.fd.flush()
            return True
        except (IOError, OSError) as e:
            if self.fd:
                self.fd.close()
                self.fd = None
            return False
    
    def release(self):
        """释放文件锁"""
        if self.fd:
            try:
                fcntl.flock(self.fd.fileno(), fcntl.LOCK_UN)
                self.fd.close()
            except Exception:
                pass
            finally:
                self.fd = None
        # 尝试删除锁文件
        try:
            if os.path.exists(self.lock_file):
                os.remove(self.lock_file)
        except Exception:
            pass
    
    def __enter__(self):
        self.acquire()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False


# ==================== GPU 检查 ====================
def check_gpu_available() -> tuple[bool, str]:
    """
    检查GPU是否可用（没有被其他程序占用）
    
    Returns:
        (是否可用, 状态信息)
    """
    try:
        if not init_gpu_monitor():
            return False, "GPU监控初始化失败"
        
        gpu_info = get_gpu_info()
        if not gpu_info:
            return False, "无法获取GPU信息"
        
        for gpu in gpu_info:
            processes = gpu.get('processes', [])
            for proc in processes:
                mem_mb = proc['mem_used_mb']
                user = proc['user']
                
                # 如果有进程占用显存超过阈值，且不是本系统用户，认为GPU被占用
                if mem_mb > GPU_MEMORY_THRESHOLD_MB:
                    # 检查是否是外部用户
                    if 'hy-worldplay' not in user.lower() and 'hunyuan' not in user.lower():
                        return False, f"GPU {gpu['id']} 被用户 {user} 占用 ({mem_mb:.0f}MB)"
        
        return True, "GPU可用"
    
    except Exception as e:
        return False, f"GPU检查出错: {str(e)}"


# ==================== 任务处理 ====================
def get_pending_task() -> Optional[Dict[str, Any]]:
    """
    获取一个待处理的任务
    
    Returns:
        任务字典，如果没有待处理任务则返回None
    """
    try:
        # 使用 MetadataManager 的 get_pending_items 方法（已包含文件锁）
        pending_items = metadata_manager.get_pending_items()
        
        if pending_items:
            item = pending_items[0]  # 获取第一个待处理任务
            return {
                "item_id": item.item_id,
                "image_id": item.image_id,
                "image_path": item.image_path,
                "prompt": item.prompt,
                "pose_id": item.pose_id,
                "pose_code": item.pose_code,
                "num_frames": item.num_frames,
                "model_type": item.model_type,
                "created_at": item.created_at,
                "seed": item.seed,
            }
        
        return None
    
    except Exception as e:
        logger.error(f"获取待处理任务失败: {e}")
        return None


def update_task_status(item_id: str, status: str, video_path: Optional[str] = None, error_msg: str = ""):
    """
    更新任务状态
    
    Args:
        item_id: 任务ID
        status: 新状态 (processing/completed/failed)
        video_path: 视频路径（完成时）
        error_msg: 错误信息（失败时）
    """
    try:
        updates = {"status": status}
        if video_path:
            updates["video_path"] = video_path
        if error_msg:
            updates["error_msg"] = error_msg
        
        metadata_manager.update_item(item_id, **updates)
        logger.info(f"任务 {item_id} 状态更新为: {status}")
    
    except Exception as e:
        logger.error(f"更新任务 {item_id} 状态失败: {e}")


def build_video_generation_command(task: Dict[str, Any], n_inference_gpu: int = 8) -> list:
    """
    构建视频生成命令
    参考 run-1.sh 中的蒸馏模型配置
    
    Args:
        task: 任务字典
        n_inference_gpu: 并行推理GPU数量
        
    Returns:
        命令列表
    """
    item_id = task["item_id"]
    image_path = task["image_path"]
    prompt = task["prompt"]
    pose_code = task["pose_code"]
    num_frames = task["num_frames"]
    # 从任务中获取种子值，默认为1
    seed = task.get("seed", SEED)
    
    # 计算宽高 (16:9 比例，480p)
    height = 480
    width = 832
    
    # 输出路径
    output_dir = os.path.join(VIDEO_DIR, item_id)
    os.makedirs(output_dir, exist_ok=True)
    
    # 构建 torchrun 命令
    cmd = [
        "torchrun",
        "--nproc_per_node", str(n_inference_gpu),
        "hyvideo/generate.py",
        "--prompt", prompt,
        "--image_path", image_path,
        "--resolution", RESOLUTION,
        "--aspect_ratio", ASPECT_RATIO,
        "--video_length", str(num_frames),
        "--seed", str(seed),
        "--rewrite", "false",
        "--sr", "false",
        "--save_pre_sr_video",
        "--pose", pose_code,
        "--output_path", output_dir,
        "--model_path", MODEL_PATH,
        "--action_ckpt", AR_DISTILL_ACTION_MODEL_PATH,
        "--few_step", "true",
        "--num_inference_steps", "4",
        "--model_type", "ar",
        "--use_vae_parallel", "false",
        "--use_sageattn", "false",
        "--use_fp8_gemm", "false",
    ]
    
    return cmd


def build_video_generation_command_debug(task: Dict[str, Any]) -> list:
    """
    【测试用】构建基于 FFmpeg 的视频生成命令
    
    用于快速测试 video_daemon 流程，无需 GPU 和模型。
    使用 FFmpeg 生成测试视频，模拟真实视频生成的输出格式。
    
    Args:
        task: 任务字典
        
    Returns:
        FFmpeg 命令列表
    """
    import shutil
    
    item_id = task["item_id"]
    image_path = task["image_path"]
    prompt = task["prompt"]
    num_frames = task.get("num_frames", 125)
    
    # 视频参数（与真实配置保持一致）
    height = 480
    width = 832
    fps = 24
    duration = num_frames / fps
    seed = task.get("seed", SEED)
    
    # 输出路径
    output_dir = os.path.join(VIDEO_DIR, item_id)
    os.makedirs(output_dir, exist_ok=True)
    
    output_video = os.path.join(output_dir, f"{item_id}_debug.mp4")
    
    # 检查 FFmpeg 是否可用
    if not shutil.which("ffmpeg"):
        logger.error("FFmpeg 未安装，请先安装 FFmpeg")
        raise RuntimeError("FFmpeg not found")
    
    # 清理 metadata 值中的特殊字符，避免 shell 解析错误
    def clean_metadata(value: str) -> str:
        """清理 metadata 字符串，移除或替换可能导致 shell 错误的字符"""
        # 替换特殊字符为安全字符
        value = value.replace('(', '[').replace(')', ']')
        value = value.replace("'", "").replace('"', '')
        value = value.replace('`', '').replace('$', '')
        value = value.replace(';', ',').replace('&', 'and')
        value = value.replace('|', '/').replace('<', '[').replace('>', ']')
        # 截断过长的字符串
        if len(value) > 200:
            value = value[:197] + "..."
        return value
    
    safe_title = clean_metadata(f"Debug Video - Seed: {seed}")
    safe_prompt = clean_metadata(prompt) if prompt else "No prompt"
    
    # 构建 FFmpeg 命令
    if os.path.exists(image_path):
        # 使用输入图片生成视频（静态图片转视频）
        cmd = [
            "ffmpeg",
            "-y",  # 覆盖输出文件
            "-loop", "1",
            "-i", image_path,  # 输入图片
            "-f", "lavfi",
            "-i", "anullsrc=r=44100:cl=stereo",  # 空音频
            "-t", str(duration),  # 持续时间
            "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,format=yuv420p,fps={fps}",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "128k",
            "-shortest",
            "-metadata", f"title={safe_title}",
            "-metadata", f"comment={safe_prompt}",
            output_video,
        ]
    else:
        # 输入图片不存在，使用测试图案生成视频
        logger.warning(f"输入图片不存在: {image_path}，使用测试图案替代")
        cmd = [
            "ffmpeg",
            "-y",  # 覆盖输出文件
            "-f", "lavfi",
            "-i", f"testsrc=duration={duration}:size={width}x{height}:rate={fps}",  # 测试图案
            "-f", "lavfi",
            "-i", f"sine=frequency=1000:duration={duration}",  # 测试音频
            "-pix_fmt", "yuv420p",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "128k",
            "-metadata", f"title={safe_title}",
            "-metadata", f"comment={safe_prompt}",
            output_video,
        ]
    
    logger.info(f"【调试模式】使用 FFmpeg 生成测试视频: {output_video}")
    return cmd


def stream_output(pipe, prefix, item_id):
    """实时读取子进程输出并写入日志"""
    try:
        for line in iter(pipe.readline, b''):
            try:
                decoded_line = line.decode('utf-8').rstrip()
            except UnicodeDecodeError:
                decoded_line = line.decode('utf-8', errors='replace').rstrip()
            if decoded_line:
                if len(decoded_line) > 0:
                    logger.info(f"[{prefix}] {decoded_line}")
    except Exception as e:
        logger.error(f"读取{prefix}时出错: {e}")
    finally:
        pipe.close()


def generate_video(task: Dict[str, Any]) -> tuple[bool, str] | None:
    """
    执行视频生成任务（实时输出日志）
    
    Args:
        task: 任务字典
        
    Returns:
        (是否成功, 结果信息) 或 None（无可用GPU，跳过本轮）
    """
    item_id = task["item_id"]
    
    try:
        # 检测可用GPU
        gpu_result = get_available_gpus()
        if gpu_result is None:
            logger.warning(f"任务 {item_id}: 没有可用的GPU，跳过本轮轮询")
            return None
        
        n_inference_gpu, cuda_devices = gpu_result
        logger.info(f"任务 {item_id} 使用 {n_inference_gpu} 个GPU: CUDA_VISIBLE_DEVICES={cuda_devices}")
        
        # 更新任务状态为处理中
        update_task_status(item_id, "processing")
        
        # 构建命令（传入动态GPU数量）
        cmd = build_video_generation_command(task, n_inference_gpu)
        logger.info(f"执行任务 {item_id}: {' '.join(cmd)}")
        
        # 设置环境变量
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = cuda_devices
        
        # 启动子进程，实时捕获输出
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=project_root,
            env=env,
            bufsize=1  # 行缓冲
        )
        
        # 创建线程实时读取stdout和stderr
        stdout_thread = threading.Thread(target=stream_output, args=(process.stdout, "OUT", item_id))
        stderr_thread = threading.Thread(target=stream_output, args=(process.stderr, "ERR", item_id))
        stdout_thread.daemon = True
        stderr_thread.daemon = True
        stdout_thread.start()
        stderr_thread.start()
        
        # 等待进程完成
        try:
            process.wait(timeout=3600)  # 1小时超时
        except subprocess.TimeoutExpired:
            process.kill()
            stdout_thread.join(timeout=5)
            stderr_thread.join(timeout=5)
            error_msg = "任务执行超时"
            logger.error(f"任务 {item_id}: {error_msg}")
            update_task_status(item_id, "failed", error_msg=error_msg)
            return False, error_msg
        
        # 等待读取线程完成
        stdout_thread.join(timeout=10)
        stderr_thread.join(timeout=10)
        
        # 检查执行结果
        if process.returncode != 0:
            error_msg = f"进程返回非零退出码: {process.returncode}"
            logger.error(f"任务 {item_id} 执行失败: {error_msg}")
            update_task_status(item_id, "failed", error_msg=error_msg)
            return False, error_msg
        
        # 查找生成的视频文件
        output_dir = os.path.join(VIDEO_DIR, item_id)
        video_files = []
        if os.path.exists(output_dir):
            for f in os.listdir(output_dir):
                if f.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    video_files.append(os.path.join(output_dir, f))
        
        if video_files:
            # 选择最新的视频文件
            video_path = max(video_files, key=os.path.getctime)
            update_task_status(item_id, "completed", video_path=video_path)
            logger.info(f"任务 {item_id} 完成，视频保存至: {video_path}")
            return True, video_path
        else:
            error_msg = "未找到生成的视频文件"
            logger.error(f"任务 {item_id}: {error_msg}")
            update_task_status(item_id, "failed", error_msg=error_msg)
            return False, error_msg
    
    except subprocess.TimeoutExpired:
        error_msg = "任务执行超时"
        logger.error(f"任务 {item_id}: {error_msg}")
        update_task_status(item_id, "failed", error_msg=error_msg)
        return False, error_msg
    
    except Exception as e:
        error_msg = str(e)
        logger.error(f"任务 {item_id} 执行异常: {error_msg}")
        update_task_status(item_id, "failed", error_msg=error_msg[:500])
        return False, error_msg


# ==================== 守护进程核心 ====================
class VideoDaemon:
    """视频生成守护进程"""
    
    def __init__(self):
        self.running = False
        self.lock = FileLock(DAEMON_LOCK_FILE)
    
    def start(self):
        """启动守护进程"""
        # 尝试获取文件锁（非阻塞）
        if not self.lock.acquire(blocking=False):
            logger.error("另一个守护进程实例已在运行")
            return False
        
        try:
            self.running = True
            
            # 重置卡住的任务
            reset_count = metadata_manager.reset_stuck_tasks()
            if reset_count > 0:
                logger.info(f"🔄 已重置 {reset_count} 个卡住的任务为 pending 状态")
            
            logger.info("=" * 60)
            logger.info("视频生成守护进程已启动")
            logger.info(f"PID: {os.getpid()}")
            logger.info(f"轮询间隔: {POLL_INTERVAL}秒")
            logger.info(f"GPU显存阈值: {GPU_MEMORY_THRESHOLD_MB}MB")
            logger.info("=" * 60)
            
            # 写入PID文件
            with open(DAEMON_PID_FILE, 'w') as f:
                f.write(str(os.getpid()))
            
            # 主循环
            while self.running:
                try:
                    self._process_cycle()
                except Exception as e:
                    logger.error(f"处理周期出错: {e}")
                
                # 等待下一次轮询
                time.sleep(POLL_INTERVAL)
        
        finally:
            self.lock.release()
            # 清理PID文件
            try:
                if os.path.exists(DAEMON_PID_FILE):
                    os.remove(DAEMON_PID_FILE)
            except Exception:
                pass
            logger.info("守护进程已停止")
    
    def _process_cycle(self):
        """单次处理周期"""
        # 1. 检查是否有待处理任务
        task = get_pending_task()
        if not task:
            return
        
        logger.info(f"发现待处理任务: {task['item_id']}")
        
        # 2. 执行视频生成（内部会检测GPU）
        result = generate_video(task)
        
        # 3. 处理结果
        if result is None:
            # 没有可用GPU，跳过本轮轮询
            logger.warning(f"任务 {task['item_id']}: 无可用GPU，等待下一轮")
            return
        
        success, message = result
        if success:
            logger.info(f"任务 {task['item_id']} 成功完成: {message}")
        else:
            logger.error(f"任务 {task['item_id']} 失败: {message}")
    
    def stop(self):
        """停止守护进程"""
        self.running = False


def daemonize():
    """
    将进程转为守护进程（Unix/Linux）
    """
    try:
        pid = os.fork()
        if pid > 0:
            # 父进程退出
            sys.exit(0)
    except OSError as e:
        logger.error(f"第一次fork失败: {e}")
        sys.exit(1)
    
    # 子进程脱离终端
    os.chdir("/")
    os.setsid()
    os.umask(0)
    
    try:
        pid = os.fork()
        if pid > 0:
            # 第一个子进程退出
            sys.exit(0)
    except OSError as e:
        logger.error(f"第二次fork失败: {e}")
        sys.exit(1)
    
    # 孙子进程成为真正的守护进程
    # 重定向标准输入输出
    sys.stdout.flush()
    sys.stderr.flush()
    
    si = open('/dev/null', 'r')
    so = open('/dev/null', 'a+')
    se = open('/dev/null', 'a+')
    
    os.dup2(si.fileno(), sys.stdin.fileno())
    os.dup2(so.fileno(), sys.stdout.fileno())
    os.dup2(se.fileno(), sys.stderr.fileno())


def stop_daemon():
    """停止正在运行的守护进程"""
    try:
        if not os.path.exists(DAEMON_PID_FILE):
            print("守护进程未运行")
            return
        
        with open(DAEMON_PID_FILE, 'r') as f:
            pid = int(f.read().strip())
        
        # 发送终止信号
        os.kill(pid, signal.SIGTERM)
        print(f"已向守护进程 (PID: {pid}) 发送停止信号")
        
        # 等待进程结束
        for _ in range(10):
            try:
                os.kill(pid, 0)
                time.sleep(0.5)
            except OSError:
                print("守护进程已停止")
                return
        
        # 强制终止
        os.kill(pid, signal.SIGKILL)
        print("已强制终止守护进程")
    
    except Exception as e:
        print(f"停止守护进程失败: {e}")


def main():
    parser = argparse.ArgumentParser(description="HY-WorldPlay 视频生成守护进程")
    parser.add_argument("--daemon", action="store_true", help="以后台守护进程模式运行")
    parser.add_argument("--stop", action="store_true", help="停止守护进程")
    parser.add_argument("--status", action="store_true", help="查看守护进程状态")
    args = parser.parse_args()
    
    if args.stop:
        stop_daemon()
        return
    
    if args.status:
        if os.path.exists(DAEMON_PID_FILE):
            with open(DAEMON_PID_FILE, 'r') as f:
                pid = f.read().strip()
            print(f"守护进程正在运行 (PID: {pid})")
            print(f"日志文件: {DAEMON_LOG_FILE}")
        else:
            print("守护进程未运行")
        return
    
    if args.daemon:
        daemonize()
    
    # 设置信号处理
    daemon = VideoDaemon()
    
    def signal_handler(signum, frame):
        logger.info(f"收到信号 {signum}，正在停止...")
        daemon.stop()
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    # 启动守护进程
    daemon.start()


if __name__ == "__main__":
    main()
