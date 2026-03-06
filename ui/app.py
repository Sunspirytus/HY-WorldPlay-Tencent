"""
HY-WorldPlay 主应用
"""

import gradio as gr
import os
import re
import subprocess
import atexit

from .config import DATA_DIR, IMAGE_DIR, POSE_DIR, VIDEO_DIR, POSE_PRESETS
from .metadata_manager import metadata_manager
from .gpu_monitor import (
    update_gpu_history,
    get_gpu_util_chart_data,
    get_gpu_memory_chart_data,
    get_gpu_users_text
)
from .utils import format_datetime

# 守护进程日志文件路径
DAEMON_LOG_FILE = os.path.join(DATA_DIR, "video_daemon.log")

# 文件锁路径
METADATA_LOCK_FILE = os.path.join(DATA_DIR, ".metadata.lock")
DAEMON_LOCK_FILE = os.path.join(DATA_DIR, "video_daemon.lock")
DAEMON_PID_FILE = os.path.join(DATA_DIR, "video_daemon.pid")


def cleanup_lock_files():
    """清理残留的文件锁，在程序启动时调用"""
    lock_files = [
        METADATA_LOCK_FILE,
        DAEMON_LOCK_FILE,
        DAEMON_PID_FILE,
    ]
    for lock_file in lock_files:
        try:
            if os.path.exists(lock_file):
                os.remove(lock_file)
                print(f"🧹 已清理残留锁文件: {os.path.basename(lock_file)}")
        except Exception as e:
            print(f"⚠️ 清理锁文件失败 {lock_file}: {e}")


def get_daemon_log_tail(lines=100):
    """获取守护进程日志的最后N行"""
    try:
        if not os.path.exists(DAEMON_LOG_FILE):
            return "日志文件不存在，守护进程可能未启动"
        
        # 使用 tail 命令获取最后N行
        result = subprocess.run(
            ["tail", "-n", str(lines), DAEMON_LOG_FILE],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            log_content = result.stdout.strip()
            if not log_content:
                return "日志文件为空"
            return log_content
        else:
            return f"读取日志失败: {result.stderr}"
    
    except subprocess.TimeoutExpired:
        return "读取日志超时"
    except Exception as e:
        return f"读取日志出错: {str(e)}"

# 守护进程管理
_daemon_process = None

def start_video_daemon():
    """启动视频生成守护进程"""
    global _daemon_process
    try:
        # 获取当前目录（ui文件夹）
        ui_dir = os.path.dirname(os.path.abspath(__file__))
        daemon_script = os.path.join(ui_dir, "video_daemon.py")
        
        if os.path.exists(daemon_script):
            # 使用 Popen 启动守护进程，不阻塞
            _daemon_process = subprocess.Popen(
                ["python3", daemon_script, "--daemon"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            print(f"🚀 视频生成守护进程已启动 (PID: {_daemon_process.pid})")
        else:
            print(f"⚠️ 守护进程脚本不存在: {daemon_script}")
    except Exception as e:
        print(f"⚠️ 启动守护进程失败: {e}")

def stop_video_daemon():
    """停止视频生成守护进程"""
    global _daemon_process
    try:
        if _daemon_process:
            _daemon_process.terminate()
            try:
                _daemon_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                _daemon_process.kill()
            print("🛑 视频生成守护进程已停止")
        
        # 同时尝试通过 PID 文件停止（以防万一）
        ui_dir = os.path.dirname(os.path.abspath(__file__))
        daemon_script = os.path.join(ui_dir, "video_daemon.py")
        if os.path.exists(daemon_script):
            subprocess.run(
                ["python3", daemon_script, "--stop"],
                capture_output=True,
                cwd=ui_dir
            )
    except Exception as e:
        print(f"⚠️ 停止守护进程时出错: {e}")


def create_ui():
    """创建Gradio界面"""
    with gr.Blocks(title="世界模型 视频生成") as app:
        
        # 状态变量
        step3_current_config = gr.State({})
        
        # ========== 顶部导航栏 ==========
        gr.Markdown("""
        # 🎮 世界模型 视频生成器
        """)
        
        with gr.Row():
            nav_step1 = gr.Button("1️⃣ 图片库", variant="primary", size="lg")
            nav_step2 = gr.Button("2️⃣ 视角库", size="lg")
            nav_step3 = gr.Button("3️⃣ 生成视频", size="lg")
            nav_step4 = gr.Button("4️⃣ 视频库", size="lg")
        
        gr.Markdown("---")
        
        # ========== 页面1: 图片库 ==========
        with gr.Column(visible=True) as page1:
            gr.Markdown("## 📤 步骤 1: 图片与描述库")
            gr.Markdown("管理所有图片和AI生成的描述。这些图片将用于视频生成。")
            
            with gr.Tabs():
                with gr.TabItem("🆕 添加新图片"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            step1_image_input = gr.Image(
                                label="上传图片",
                                type="filepath",
                                height=350,
                                sources=["upload"]
                            )
                            step1_name_input = gr.Textbox(
                                label="图片名称",
                                placeholder="给这张图片起个名字（必填，不可重复）",
                                interactive=True
                            )
                            step1_language = gr.Radio(
                                choices=[("中文", "zh"), ("English", "en")],
                                value="zh",
                                label="描述语言"
                            )
                            step1_generate_btn = gr.Button("📝 生成描述", variant="primary")
                        
                        with gr.Column(scale=2):
                            step1_prompt = gr.Textbox(
                                label="场景描述（可编辑）",
                                lines=10,
                                interactive=True
                            )
                            step1_status = gr.Textbox(label="状态", interactive=False)
                            step1_save_btn = gr.Button("💾 保存到图片库", variant="primary", interactive=False)
                
                with gr.TabItem("📚 图片库"):
                    with gr.Row():
                        with gr.Column(scale=3):
                            with gr.Row():
                                step1_gallery = gr.Gallery(
                                    label="点击图片查看",
                                    columns=5,
                                    rows=3,
                                    object_fit="cover"
                                )
                            
                            with gr.Row():
                                step1_refresh_gallery = gr.Button("🔄 刷新", scale=10)
                                step1_delete_item = gr.Button("🗑️ 删除选中", variant="stop", scale=1)

                        with gr.Column(scale=1):
                            step1_selected_info = gr.JSON(label="选中项目信息")
        
        # ========== 页面2: 视角库 ==========
        with gr.Column(visible=True) as page2:
            gr.Markdown("## 🎥 步骤 2: 视角轨迹库")
            gr.Markdown("创建和管理相机视角轨迹。视角轨迹与图片独立，可在生成视频时灵活组合。")
            
            with gr.Tabs():
                with gr.TabItem("🆕 创建新视角"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### 视角名称")
                            step2_name = gr.Textbox(
                                label="视角名称",
                                placeholder="给这个视角起个名字，如：向前探索"
                            )

                            gr.Markdown("### 轨迹步数")
                            step2_num_steps = gr.Slider(
                                minimum=5, maximum=65, step=4, value=31,
                                label="轨迹步数",
                                interactive=True,
                            )
                            
                            gr.Markdown("### 视频帧数")
                            step2_num_frames = gr.Number(
                                label="视频帧数 (步数 × 4 + 1)",
                                value=125,  # 31 * 4 + 1
                                interactive=False,
                            )
                            
                            gr.Markdown("### 快速预设")
                            step2_preset = gr.Dropdown(
                                choices=list(POSE_PRESETS.keys()),
                                value=list(POSE_PRESETS.keys())[0] if POSE_PRESETS else None,
                                label="选择预设"
                            )
                        
                        with gr.Column(scale=2):
                            gr.Markdown("### 轨迹代码")
                            step2_pose_code = gr.Textbox(
                                label="轨迹代码（可从预设自动填充，也可手动编辑）",
                                placeholder="例如: w-10, right-5, w-10",
                                lines=3
                            )
                            gr.Markdown("""
                            **格式说明**: `动作-持续时间`
                            - 移动: `w`(前), `s`(后), `a`(左), `d`(右)
                            - 旋转: `up`(抬头), `down`(低头), `left`(左转), `right`(右转)
                            - 组合: 用逗号分隔，如 `w-15, right-15, w-15`
                            """)
                            
                            step2_status = gr.Textbox(
                                label="状态 / 轨迹帧数信息",
                                interactive=False,
                                value="请输入轨迹代码"
                            )
                            
                            step2_save_btn = gr.Button("💾 保存到视角库", variant="primary")
                
                with gr.TabItem("📚 视角库"):
                    with gr.Row():
                        step2_list = gr.Dataframe(
                            headers=["ID", "名称", "轨迹代码", "帧数", "创建时间"],
                            label="已保存的视角",
                            interactive=False
                        )
                    
                    with gr.Row():
                        step2_refresh_list = gr.Button("🔄 刷新", scale=10)
                        step2_delete_pose = gr.Button("🗑️ 删除选中", variant="stop", scale=1)
                    
                    step2_selected = gr.Textbox(label="选中视角ID", interactive=False)
        
        # ========== 页面3: 生成视频 ==========
        with gr.Column(visible=True) as page3:
            gr.Markdown("## ⚙️ 步骤 3: 生成视频")
            gr.Markdown("从图片库和视角库中选择组合，提交视频生成任务。")
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 选择素材")
                    
                    step3_image_dropdown = gr.Dropdown(
                        label="📷 选择图片",
                        choices=[],
                        interactive=True
                    )
                    step3_refresh_images_btn = gr.Button("🔄 刷新图片列表", size="sm")
                    
                    step3_image_preview = gr.Image(
                        label="图片预览",
                        interactive=False,
                        height=200
                    )
                    step3_prompt_display = gr.Textbox(
                        label="图片描述",
                        interactive=False,
                        lines=3
                    )

                with gr.Column(scale=1): 
                    gr.Markdown("### 选择视角")
                    
                    step3_pose_dropdown = gr.Dropdown(
                        label="🎥 选择视角",
                        choices=[],
                        interactive=True
                    )
                    step3_refresh_poses_btn = gr.Button("🔄 刷新视角列表", size="sm")
                    
                    step3_pose_display = gr.Textbox(
                        label="视角轨迹",
                        interactive=False,
                        lines=2
                    )
                    step3_frames_display = gr.Number(
                        label="帧数",
                        interactive=False
                    )
                
                with gr.Column(scale=1):
                    gr.Markdown("### 生成配置")
                    
                    step3_config = gr.JSON(label="当前配置")
                    
                    step3_model = gr.Dropdown(
                        choices=[("蒸馏模型 (速度更快)", "distill")],
                        value="distill",
                        label="模型类型"
                    )
                    
                    step3_seed = gr.Number(
                        label="🎲 随机种子",
                        value=1,
                        minimum=0,
                        maximum=2147483647,
                        step=1,
                        info="设置随机种子以获得可复现的结果，默认值为1"
                    )
                    
                    gr.Markdown("""
                    **预估信息**:
                    - 生成时间: 约 5-10 分钟
                    - 输出分辨率: 480P
                    - 输出格式: MP4
                    """)
                    
                    step3_submit = gr.Button("🚀 提交生成任务", variant="primary", size="lg", interactive=False)
                    step3_status = gr.Textbox(label="状态", interactive=False)
        
        # ========== 页面4: 视频库 ==========
        with gr.Column(visible=True) as page4:
            gr.Markdown("## 🎬 步骤 4: 视频库")
            gr.Markdown("查看和管理所有生成的视频，监控GPU状态。")
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 🖥️ GPU监控")
                    
                    step4_gpu_util_chart = gr.LinePlot(
                        label="📊 GPU利用率趋势",
                        x="时间",
                        y="利用率(%)",
                        color="GPU",
                        y_lim=[0, 100],
                        height=200,
                        show_label=True
                    )
                    
                    step4_gpu_mem_chart = gr.LinePlot(
                        label="💾 显存使用趋势",
                        x="时间",
                        y="显存(GB)",
                        color="GPU",
                        height=200,
                        show_label=True
                    )
                    
                    step4_gpu_users = gr.Textbox(
                        value="",
                        label="👥 GPU占用用户",
                        interactive=False,
                        lines=4
                    )
                
                with gr.Column(scale=2):
                    gr.Markdown("### 📹 视频管理")
                    
                    with gr.Tabs():
                        with gr.TabItem("🔴 正在生成"):
                            with gr.Row():
                                step4_processing_list = gr.Dropdown(
                                    label="进行中的任务",
                                    choices=[]
                                )
                                step4_refresh_processing_btn = gr.Button("🔄 刷新")
                            
                            step4_process_info = gr.JSON(label="任务详情")
                        
                        with gr.TabItem("❌ 失败任务"):
                            with gr.Row():
                                step4_failed_list = gr.Dropdown(
                                    label="失败的任务",
                                    choices=[]
                                )
                                step4_refresh_failed_btn = gr.Button("🔄 刷新")
                                step4_retry_failed_btn = gr.Button("🔄 重试任务", variant="primary")
                            
                            with gr.Row():
                                with gr.Column(scale=1):
                                    step4_failed_info = gr.JSON(label="任务信息")
                                
                                with gr.Column(scale=2):
                                    step4_failed_error = gr.Textbox(
                                        label="错误日志",
                                        lines=10,
                                        interactive=False,
                                        max_lines=15
                                    )
                            
                            step4_delete_failed_btn = gr.Button("🗑️ 删除失败任务", variant="stop")
                        
                        with gr.TabItem("✅ 已完成"):
                            with gr.Row():
                                with gr.Column(scale=1):
                                    step4_completed_list = gr.Dropdown(
                                        label="选择视频",
                                        choices=[]
                                    )
                                    step4_refresh_completed_btn = gr.Button("🔄 刷新")
                                    step4_video_info = gr.JSON(label="视频信息")
                                    step4_delete_video = gr.Button("🗑️ 删除", variant="stop")
                                
                                with gr.Column(scale=2):
                                    step4_video_player = gr.Video(
                                        label="视频预览",
                                        height=400,
                                        loop=True,
                                        autoplay=True,
                                    )

                gr.Markdown("### 📜 守护进程日志")
                step4_daemon_log = gr.Textbox(
                    value="",
                    label="video_daemon 日志 (每5秒刷新)",
                    interactive=False,
                    lines=15,
                    max_lines=15,
                    autoscroll=True
                )
        
        # ========== 导航切换 ==========
        def switch_page(page_name: str):
            return {
                page1: gr.Column(visible=(page_name == "step1")),
                page2: gr.Column(visible=(page_name == "step2")),
                page3: gr.Column(visible=(page_name == "step3")),
                page4: gr.Column(visible=(page_name == "step4")),
                nav_step1: gr.Button(variant="primary" if page_name == "step1" else "secondary"),
                nav_step2: gr.Button(variant="primary" if page_name == "step2" else "secondary"),
                nav_step3: gr.Button(variant="primary" if page_name == "step3" else "secondary"),
                nav_step4: gr.Button(variant="primary" if page_name == "step4" else "secondary"),
            }
        
        nav_step1.click(fn=lambda: switch_page("step1"),
                       outputs=[page1, page2, page3, page4,
                               nav_step1, nav_step2, nav_step3, nav_step4])
        nav_step2.click(fn=lambda: switch_page("step2"),
                       outputs=[page1, page2, page3, page4,
                               nav_step1, nav_step2, nav_step3, nav_step4])
        nav_step3.click(fn=lambda: switch_page("step3"),
                       outputs=[page1, page2, page3, page4,
                               nav_step1, nav_step2, nav_step3, nav_step4])
        nav_step4.click(fn=lambda: switch_page("step4"),
                       outputs=[page1, page2, page3, page4,
                               nav_step1, nav_step2, nav_step3, nav_step4])
        
        # ========== 页面1 事件 ==========
        def step1_generate(image, language):
            from .utils import generate_prompt_from_image
            if not image:
                return "", "❌ 请先上传图片", gr.Button(interactive=False)
            prompt = generate_prompt_from_image(image, language)
            if prompt.startswith("错误") or prompt.startswith("API"):
                return "", f"❌ {prompt}", gr.Button(interactive=False)
            return prompt, "✅ 描述已生成", gr.Button(interactive=True)
        
        def step1_save(image, name, prompt):
            if not image:
                return "❌ 请先上传图片", gr.Gallery(), gr.Textbox(value="")
            if not name or not name.strip():
                return "❌ 请输入图片名称", gr.Gallery(), gr.Textbox(value="")
            try:
                img = metadata_manager.add_image(image, prompt, name.strip())
                display_name = f"{img.name} ({img.image_id})" if img.name else img.image_id
                return f"✅ 已保存: {display_name}", get_image_gallery(), gr.Textbox(value="")
            except ValueError as e:
                return f"❌ {str(e)}", gr.Gallery(), gr.Textbox(value=name)
        
        def get_image_gallery():
            images = metadata_manager.get_all_images()
            result = []
            for img in images:
                if os.path.exists(img.image_path):
                    # 显示格式：名称 (ID) 或 ID（如果没有名称）
                    display_name = f"{img.name} ({img.image_id})" if img.name else img.image_id
                    caption = f"{display_name}\n{format_datetime(img.created_at)}"
                    result.append((img.image_path, caption))
            return result
        
        step1_generate_btn.click(
            fn=step1_generate,
            inputs=[step1_image_input, step1_language],
            outputs=[step1_prompt, step1_status, step1_save_btn]
        )
        
        step1_save_btn.click(fn=step1_save, inputs=[step1_image_input, step1_name_input, step1_prompt],
                            outputs=[step1_status, step1_gallery, step1_name_input])
        step1_refresh_gallery.click(fn=get_image_gallery, outputs=[step1_gallery])
        
        def step1_on_select(evt: gr.SelectData):
            if isinstance(evt.value, dict):
                caption = evt.value.get("caption", "")
            elif isinstance(evt.value, tuple):
                caption = evt.value[1] if len(evt.value) > 1 else str(evt.value[0])
            else:
                caption = str(evt.value)
            # 从 caption 中提取 ID（格式可能是 "名称 (ID)" 或 "ID"）
            first_line = caption.split("\n")[0].strip()
            # 提取括号中的 ID 或直接作为 ID
            if "(" in first_line and ")" in first_line:
                image_id = first_line.split("(")[1].split(")")[0].strip()
            else:
                image_id = first_line
            img = metadata_manager.get_image(image_id)
            if img:
                return {
                    "name": img.name,
                    "image_id": image_id,
                    "prompt": img.prompt,
                    "created": format_datetime(img.created_at)
                }
            return {}
        
        step1_gallery.select(fn=step1_on_select, outputs=[step1_selected_info])
        
        def step1_delete(info):
            if info and "image_id" in info:
                metadata_manager.delete_image(info["image_id"])
                return get_image_gallery(), {}
            return get_image_gallery(), {}
        
        step1_delete_item.click(fn=step1_delete, inputs=[step1_selected_info],
                               outputs=[step1_gallery, step1_selected_info])
        
        # ========== 页面2 事件 ==========
        from .utils import calculate_pose_frames, validate_pose_code
        
        def step2_preset_change(preset, num_steps):
            """选择预设时，根据当前步数自动计算并调整轨迹代码
            
            横杠后面的值就是步数，总步数之和等于设置的num_steps
            帧数仅用于状态栏提示: frames = steps * 4 + 1
            """
            if not preset:
                return ""
            
            base_code = POSE_PRESETS.get(preset, "")
            if not base_code:
                return ""
            
            # 目标步数就是用户选择的步数
            target_steps = num_steps
            
            # 解析基础轨迹代码，获取原始步数（横杠后面的值）
            actions = [a.strip() for a in base_code.split(',')]
            base_steps_list = []
            for a in actions:
                match = re.match(r'^([a-z]+)-(\d+)$', a)
                if match:
                    step_count = int(match.group(2))  # 横杠后面直接是步数
                    base_steps_list.append((match.group(1), step_count))
            
            total_base_steps = sum(steps for _, steps in base_steps_list)
            
            if total_base_steps == 0:
                # 如果无法解析，直接生成单动作轨迹
                return f"w-{target_steps}"
            
            # 按比例分配步数到每个动作
            scaled_actions = []
            remaining_steps = target_steps
            
            for i, (action_name, base_step) in enumerate(base_steps_list):
                if i == len(base_steps_list) - 1:
                    # 最后一个动作，使用剩余的步数
                    allocated_steps = remaining_steps
                else:
                    # 按比例分配步数
                    allocated_steps = max(1, round(target_steps * base_step / total_base_steps))
                    # 确保至少为每个剩余动作保留1步
                    min_remaining = len(base_steps_list) - i - 1
                    allocated_steps = min(allocated_steps, remaining_steps - min_remaining)
                
                scaled_actions.append(f"{action_name}-{allocated_steps}")
                remaining_steps -= allocated_steps
            
            return ", ".join(scaled_actions)
        
        def step2_update_pose_steps_info(pose_code, num_steps):
            """根据步数验证轨迹代码，只计算步数总和"""
            if not pose_code or not pose_code.strip():
                return "请输入轨迹代码"
            is_valid, error_msg = validate_pose_code(pose_code)
            if not is_valid:
                return f"❌ {error_msg}"
            
            # 计算轨迹代码中的总步数（横杠后面的值相加）
            actions = [a.strip() for a in pose_code.split(',')]
            total_steps = 0
            for action in actions:
                match = re.match(r'^([a-z]+)-(\d+)$', action)
                if match:
                    total_steps += int(match.group(2))
            
            if total_steps > num_steps:
                return f"❌ 轨迹步数({total_steps})大于设置的步数({num_steps})，请减少轨迹动作或增加步数"
            elif total_steps < num_steps:
                remaining = num_steps - total_steps
                return f"✅ 当前轨迹步数: {total_steps}，还差{remaining}步将自动填充（对应帧数: {num_steps * 4 + 1}）"
            else:
                return f"✅ 轨迹步数: {total_steps}，与设置匹配（对应帧数: {num_steps * 4 + 1}）"
     
        def step2_save(name, pose_code, num_steps):
            """保存视角，只保存步数，不保存帧数"""
            if not name or not pose_code:
                return "❌ 请输入名称和轨迹代码"
            is_valid, error_msg = validate_pose_code(pose_code)
            if not is_valid:
                return f"❌ 轨迹代码格式错误: {error_msg}"
            
            # 计算轨迹代码中的总步数
            actions = [a.strip() for a in pose_code.split(',')]
            total_steps = 0
            for action in actions:
                match = re.match(r'^([a-z]+)-(\d+)$', action)
                if match:
                    total_steps += int(match.group(2))
            
            if total_steps > num_steps:
                return f"❌ 轨迹步数({total_steps})大于设置的步数({num_steps})，请减少轨迹动作或增加步数"
            
            final_pose_code = pose_code
            if total_steps < num_steps:
                remaining = num_steps - total_steps
                final_pose_code = f"{pose_code}, w-{remaining}"
            
            # 只保存步数，不保存帧数
            full_name = f"{name}-{num_steps}"
            existing_poses = metadata_manager.get_all_poses()
            for p in existing_poses:
                if p.name == full_name:
                    return f"❌ 视角名称 '{full_name}' 已存在，请使用其他名称"
            
            # 计算帧数仅用于提示用户
            num_frames = num_steps * 4 + 1
            pose = metadata_manager.add_pose(full_name, final_pose_code, num_frames)
            if total_steps < num_steps:
                return f"✅ 已保存视角: {full_name} (ID: {pose.pose_id})，轨迹已自动填充至{num_steps}步（对应帧数: {num_frames}）"
            return f"✅ 已保存视角: {full_name} (ID: {pose.pose_id})（对应帧数: {num_frames}）"
        
        def get_pose_list():
            poses = metadata_manager.get_all_poses()
            return [[p.pose_id, p.name, p.pose_code, p.num_frames, format_datetime(p.created_at)] for p in poses]
        
        step2_preset.change(fn=step2_preset_change, inputs=[step2_preset, step2_num_steps], outputs=[step2_pose_code])
        step2_pose_code.change(fn=step2_update_pose_steps_info, inputs=[step2_pose_code, step2_num_steps], outputs=[step2_status])
        
        def step2_on_steps_change(num_steps):
            """修改步数时更新帧数显示、清空轨迹代码和状态"""
            num_frames = num_steps * 4 + 1
            return num_frames, "", f"步数已改为 {num_steps}（对应帧数: {num_frames}），请重新选择预设或输入轨迹代码"
        
        step2_num_steps.change(fn=step2_on_steps_change, inputs=[step2_num_steps], outputs=[step2_num_frames, step2_pose_code, step2_status])
        step2_save_btn.click(fn=step2_save, inputs=[step2_name, step2_pose_code, step2_num_steps], outputs=[step2_status])
        step2_refresh_list.click(fn=lambda: (gr.Dataframe(value=get_pose_list()), ""), outputs=[step2_list, step2_selected])
        
        def step2_on_select(evt: gr.SelectData):
            row = evt.index[0] if evt.index else 0
            poses = metadata_manager.get_all_poses()
            if row < len(poses):
                return poses[row].pose_id
            return ""
        
        step2_list.select(fn=step2_on_select, outputs=[step2_selected])
        
        def step2_delete(pose_id):
            if pose_id:
                metadata_manager.delete_pose(pose_id)
            return get_pose_list(), ""
        
        step2_delete_pose.click(fn=step2_delete, inputs=[step2_selected], outputs=[step2_list, step2_selected])
        
        # ========== 页面3 事件 ==========
        def step3_refresh_images():
            images = metadata_manager.get_all_images()
            choices = [(f"{img.name}-{img.image_id} - {img.prompt if len(img.prompt) <= 30 else img.prompt[:30] + '...'}", img.image_id) for img in images]
            return gr.Dropdown(choices=choices or [("无图片", "")])
        
        def step3_refresh_poses():
            poses = metadata_manager.get_all_poses()
            choices = [(f"{pose.name} - {pose.pose_code if len(pose.pose_code) <= 30 else pose.pose_code[:30] + '...'}", pose.pose_id) for pose in poses]
            return gr.Dropdown(choices=choices or [("无视角", "")])
        
        nav_step3.click(fn=step3_refresh_images, outputs=[step3_image_dropdown])
        nav_step3.click(fn=step3_refresh_poses, outputs=[step3_pose_dropdown])
        step3_refresh_images_btn.click(fn=step3_refresh_images, outputs=[step3_image_dropdown])
        step3_refresh_poses_btn.click(fn=step3_refresh_poses, outputs=[step3_pose_dropdown])
        
        def step3_on_image_change(image_id, current_config):
            config = dict(current_config) if current_config else {}
            if not image_id:
                config.pop("image_id", None)
                config.pop("prompt", None)
                return None, "", config, config
            img = metadata_manager.get_image(image_id)
            if not img:
                config.pop("image_id", None)
                config.pop("prompt", None)
                return None, "", config, config
            config["image_id"] = image_id
            config["prompt"] = img.prompt[:50]
            return img.image_path, img.prompt, config, config
        
        def step3_on_pose_change(pose_id, current_config):
            config = dict(current_config) if current_config else {}
            if not pose_id:
                config.pop("pose_id", None)
                config.pop("pose_name", None)
                config.pop("pose_code", None)
                config.pop("num_frames", None)
                return "", 0, config, config
            pose = metadata_manager.get_pose(pose_id)
            if not pose:
                config.pop("pose_id", None)
                config.pop("pose_name", None)
                config.pop("pose_code", None)
                config.pop("num_frames", None)
                return "", 0, config, config
            config["pose_id"] = pose_id
            config["pose_name"] = pose.name
            config["pose_code"] = pose.pose_code
            config["num_frames"] = pose.num_frames
            return pose.pose_code, pose.num_frames, config, config
        
        step3_image_dropdown.change(fn=step3_on_image_change, inputs=[step3_image_dropdown, step3_config],
                                   outputs=[step3_image_preview, step3_prompt_display, step3_config, step3_config])
        step3_pose_dropdown.change(fn=step3_on_pose_change, inputs=[step3_pose_dropdown, step3_config],
                                  outputs=[step3_pose_display, step3_frames_display, step3_config, step3_config])
        
        def step3_check_submit(image_id, pose_id):
            return gr.Button(interactive=bool(image_id and pose_id))
        
        step3_image_dropdown.change(fn=step3_check_submit, inputs=[step3_image_dropdown, step3_pose_dropdown], outputs=[step3_submit])
        step3_pose_dropdown.change(fn=step3_check_submit, inputs=[step3_image_dropdown, step3_pose_dropdown], outputs=[step3_submit])
        
        def step3_submit_task(image_id, pose_id, model_type, seed):
            if not image_id or not pose_id:
                return "❌ 请选择图片和视角"
            # 确保 seed 是整数
            try:
                seed = int(seed) if seed else 1
            except (ValueError, TypeError):
                seed = 1
            item = metadata_manager.add_item(image_id, pose_id, model_type, seed=seed)
            if not item:
                return "❌ 创建任务失败"
            return f"✅ 已提交任务: {item.item_id} (种子: {item.seed})"
        
        step3_submit.click(fn=step3_submit_task, inputs=[step3_image_dropdown, step3_pose_dropdown, step3_model, step3_seed], outputs=[step3_status])
        
        # ========== 页面4 事件 ==========
        def step4_refresh():
            items = metadata_manager.get_all_items()
            processing = [(f"{metadata_manager.get_image(i.image_id).name}-{metadata_manager.get_pose(i.pose_id).name} - {i.item_id} - {i.status}", i.item_id) for i in items if i.status in ("pending", "processing")]
            failed = [(f"{metadata_manager.get_image(i.image_id).name}-{metadata_manager.get_pose(i.pose_id).name} - {i.item_id} - 创建于 {format_datetime(i.created_at)}", i.item_id) for i in items if i.status == "failed"]
            completed = [(f"{metadata_manager.get_image(i.image_id).name}-{metadata_manager.get_pose(i.pose_id).name} - {i.item_id} - {format_datetime(i.created_at)}", i.item_id) for i in items if i.video_path]
            return (
                gr.Dropdown(choices=processing or [("无进行中任务", "")]),
                gr.Dropdown(choices=failed or [("无失败任务", "")]),
                gr.Dropdown(choices=completed or [("无已完成视频", "")])
            )
        
        nav_step4.click(fn=step4_refresh, outputs=[step4_processing_list, step4_failed_list, step4_completed_list])
        step4_refresh_processing_btn.click(fn=step4_refresh, outputs=[step4_processing_list, step4_failed_list, step4_completed_list])
        step4_refresh_failed_btn.click(fn=step4_refresh, outputs=[step4_processing_list, step4_failed_list, step4_completed_list])
        step4_refresh_completed_btn.click(fn=step4_refresh, outputs=[step4_processing_list, step4_failed_list, step4_completed_list])
        
        def step4_load_failed(item_id):
            if not item_id:
                return {}, ""
            item = metadata_manager.get_item(item_id)
            if item:
                info = {
                    "item_id": item.item_id,
                    "image_id": item.image_id,
                    "pose_id": item.pose_id,
                    "status": item.status,
                    "created": format_datetime(item.created_at),
                }
                return info, item.error_msg or "无错误信息"
            return {}, ""
        
        step4_failed_list.change(fn=step4_load_failed, inputs=[step4_failed_list], outputs=[step4_failed_info, step4_failed_error])
        
        def step4_retry_failed(item_id):
            if not item_id:
                return "❌ 请先选择要重试的任务", gr.Dropdown(), gr.Dropdown(), gr.Dropdown()
            item = metadata_manager.get_item(item_id)
            if not item:
                return "❌ 任务不存在", gr.Dropdown(), gr.Dropdown(), gr.Dropdown()
            if item.status != "failed":
                return f"❌ 任务状态为 {item.status}，不是失败任务", gr.Dropdown(), gr.Dropdown(), gr.Dropdown()
            # 重置任务状态为 pending
            metadata_manager.update_item(item_id, status="pending", error_msg="")
            return f"✅ 任务 {item_id} 已重置为待处理状态", *step4_refresh()
        
        step4_retry_failed_btn.click(fn=step4_retry_failed, inputs=[step4_failed_list], 
                                     outputs=[step3_status, step4_processing_list, step4_failed_list, step4_completed_list])
        
        def step4_delete_failed(item_id):
            if item_id:
                metadata_manager.delete_item(item_id)
            return step4_refresh()
        
        step4_delete_failed_btn.click(fn=step4_delete_failed, inputs=[step4_failed_list], 
                                      outputs=[step4_processing_list, step4_failed_list, step4_completed_list])
        
        def step4_load_processing(item_id):
            if not item_id:
                return {}
            item = metadata_manager.get_item(item_id)
            return item.to_dict() if item else {}
        
        step4_processing_list.change(fn=step4_load_processing, inputs=[step4_processing_list], outputs=[step4_process_info])
        
        def step4_do_load_video(item_id):
            if not item_id:
                return None, {}
            item = metadata_manager.get_item(item_id)
            if item and item.video_path and os.path.exists(item.video_path):
                return item.video_path, item.to_dict()
            return None, {"error": "视频不存在"}
        
        # 选择视频后自动加载
        step4_completed_list.change(fn=step4_do_load_video, inputs=[step4_completed_list], outputs=[step4_video_player, step4_video_info])
        
        def step4_delete(item_id):
            if item_id:
                metadata_manager.delete_item(item_id)
            return step4_refresh()
        
        step4_delete_video.click(fn=step4_delete, inputs=[step4_completed_list], outputs=[step4_processing_list, step4_completed_list])
        
        def step4_refresh_gpu_charts():
            update_gpu_history()
            util_data = get_gpu_util_chart_data()
            mem_data = get_gpu_memory_chart_data()
            users_text = get_gpu_users_text()
            return util_data, mem_data, users_text
        
        nav_step4.click(fn=step4_refresh_gpu_charts, outputs=[step4_gpu_util_chart, step4_gpu_mem_chart, step4_gpu_users])
        
        # GPU 监控定时刷新（每5秒）
        gpu_timer = gr.Timer(5, active=True)
        gpu_timer.tick(fn=step4_refresh_gpu_charts, outputs=[step4_gpu_util_chart, step4_gpu_mem_chart, step4_gpu_users])
        
        # 守护进程日志定时刷新（每5秒）
        log_timer = gr.Timer(5, active=True)
        log_timer.tick(fn=get_daemon_log_tail, outputs=[step4_daemon_log])
        
        # ========== 全局加载 ==========
        app.load(fn=get_image_gallery, outputs=[step1_gallery])
        app.load(fn=lambda: gr.Dataframe(value=get_pose_list()), outputs=[step2_list])
        app.load(fn=step4_refresh_gpu_charts, outputs=[step4_gpu_util_chart, step4_gpu_mem_chart, step4_gpu_users])
        
        def init_page():
            return {
                page1: gr.Column(visible=True),
                page2: gr.Column(visible=False),
                page3: gr.Column(visible=False),
                page4: gr.Column(visible=False),
                nav_step1: gr.Button(variant="primary"),
                nav_step2: gr.Button(variant="secondary"),
                nav_step3: gr.Button(variant="secondary"),
                nav_step4: gr.Button(variant="secondary"),
            }
        
        app.load(fn=init_page, outputs=[page1, page2, page3, page4, nav_step1, nav_step2, nav_step3, nav_step4])
    
    return app


def main():
    """主函数"""
    # 清理残留的文件锁
    cleanup_lock_files()
    
    print(f"📁 数据存储目录: {os.path.abspath(DATA_DIR)}")
    print(f"🖼️  图片存储目录: {os.path.abspath(IMAGE_DIR)}")
    print(f"🎥 视角存储目录: {os.path.abspath(POSE_DIR)}")
    print(f"🎬 视频存储目录: {os.path.abspath(VIDEO_DIR)}")
    
    # 启动视频生成守护进程
    start_video_daemon()
    
    # 注册退出时停止守护进程
    atexit.register(stop_video_daemon)
    
    try:
        app = create_ui()
        app.launch(server_name="0.0.0.0", server_port=7860, show_error=True, root_path="/infer")
    finally:
        # 确保守护进程被停止
        stop_video_daemon()


if __name__ == "__main__":
    main()