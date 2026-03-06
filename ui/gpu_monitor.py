"""
HY-WorldPlay GPU 监控模块
"""

import os
import sys
import time
from datetime import datetime
from typing import List, Dict, Tuple

# 添加 trainer 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'trainer'))
from third_party.pynvml import *

_gpu_monitor_initialized = False


def init_gpu_monitor():
    """初始化GPU监控"""
    global _gpu_monitor_initialized
    if not _gpu_monitor_initialized:
        try:
            nvmlInit()
            _gpu_monitor_initialized = True
        except Exception as e:
            print(f"GPU监控初始化失败: {e}")
    return _gpu_monitor_initialized


def get_gpu_processes_info(gpu_handle):
    """获取GPU上的进程信息"""
    processes_info = []
    try:
        processes = nvmlDeviceGetComputeRunningProcesses(gpu_handle)
        for proc in processes:
            pid = proc.pid
            mem_used = proc.usedGpuMemory / (1024 ** 2)  # MB
            try:
                import pwd
                proc_path = f"/proc/{pid}"
                if os.path.exists(proc_path):
                    stat_info = os.stat(proc_path)
                    try:
                        user = pwd.getpwuid(stat_info.st_uid).pw_name
                    except KeyError:
                        user = f"uid:{stat_info.st_uid}"
                else:
                    user = "unknown"
            except Exception:
                user = "unknown"
            
            processes_info.append({
                "pid": pid,
                "user": user,
                "mem_used_mb": mem_used
            })
    except Exception as e:
        pass
    return processes_info


def get_gpu_info():
    """获取GPU信息"""
    gpu_info = []
    try:
        if not _gpu_monitor_initialized:
            init_gpu_monitor()
        
        if _gpu_monitor_initialized:
            device_count = nvmlDeviceGetCount()
            for i in range(device_count):
                handle = nvmlDeviceGetHandleByIndex(i)
                name = nvmlDeviceGetName(handle)
                
                # GPU利用率
                util = nvmlDeviceGetUtilizationRates(handle)
                gpu_util = util.gpu
                mem_util = util.memory
                
                # 显存信息
                mem_info = nvmlDeviceGetMemoryInfo(handle)
                mem_used = mem_info.used / (1024 ** 3)  # GB
                mem_total = mem_info.total / (1024 ** 3)  # GB
                
                # 温度
                try:
                    temp = nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)
                except:
                    temp = "N/A"
                
                # 功耗
                try:
                    power = nvmlDeviceGetPowerUsage(handle) / 1000  # W
                except:
                    power = "N/A"
                
                # 进程信息
                processes = get_gpu_processes_info(handle)
                
                gpu_info.append({
                    "id": i,
                    "name": name,
                    "gpu_util": gpu_util,
                    "mem_util": mem_util,
                    "mem_used": f"{mem_used:.2f}",
                    "mem_total": f"{mem_total:.2f}",
                    "temperature": temp,
                    "power": f"{power:.1f}" if isinstance(power, float) else power,
                    "processes": processes
                })
    except Exception as e:
        print(f"获取GPU信息失败: {e}")
    
    return gpu_info


def check_gpu_status():
    """
    检查GPU状态
    如果所有显卡的显存占用大于500M，并且有非hy-worldplay的用户占用GPU，状态就显示忙碌
    返回: (status_text, status_emoji, is_busy, all_gpu_users)
    """
    gpu_info = get_gpu_info()
    if not gpu_info:
        return "❌ 无法获取GPU信息", "❌", True, []
    
    all_users = set()
    has_high_mem_usage = False
    has_other_users = False
    
    for gpu in gpu_info:
        for proc in gpu.get('processes', []):
            user = proc['user']
            mem_mb = proc['mem_used_mb']
            all_users.add(user)
            
            # 检查是否有显存占用大于500M的进程
            if mem_mb > 500:
                has_high_mem_usage = True
                # 检查是否是非hy-worldplay用户
                if 'hy-worldplay' not in user.lower() and 'hunyuan' not in user.lower():
                    has_other_users = True
    
    if has_high_mem_usage and has_other_users:
        return "🔴 忙碌", "🔴", True, list(all_users)
    elif has_high_mem_usage:
        return "🟡 占用中", "🟡", False, list(all_users)
    else:
        return "🟢 空闲", "🟢", False, list(all_users)


def format_gpu_display():
    """格式化GPU显示信息"""
    gpu_info = get_gpu_info()
    if not gpu_info:
        return "❌ 无法获取GPU信息"
    
    result = []
    for gpu in gpu_info:
        # 格式化进程信息
        proc_info = ""
        processes = gpu.get('processes', [])
        if processes:
            proc_lines = []
            for proc in processes:
                proc_lines.append(f"    - PID {proc['pid']}: {proc['user']} ({proc['mem_used_mb']:.0f}MB)")
            proc_info = "\n".join(proc_lines)
        else:
            proc_info = "    无进程"
        
        result.append(f"""**GPU {gpu['id']}: {gpu['name']}**
- 🎮 GPU利用率: **{gpu['gpu_util']}%**
- 💾 显存利用率: **{gpu['mem_util']}%** ({gpu['mem_used']}/{gpu['mem_total']} GB)
- 🌡️ 温度: **{gpu['temperature']}°C**
- ⚡ 功耗: **{gpu['power']}W**
- 📋 进程信息:
{proc_info}
---""")
    
    return "\n".join(result)


def format_gpu_status_panel():
    """格式化GPU状态面板"""
    status_text, status_emoji, is_busy, all_users = check_gpu_status()
    
    gpu_info = get_gpu_info()
    
    # 构建用户列表显示
    user_display = ""
    if all_users:
        user_display = "\n".join([f"  - {u}" for u in sorted(all_users)])
    else:
        user_display = "  无活跃用户"
    
    # 构建GPU简要状态
    gpu_summary = ""
    for gpu in gpu_info:
        gpu_summary += f"GPU{gpu['id']}: {gpu['gpu_util']}% | {gpu['mem_used']}/{gpu['mem_total']}GB\n"
    
    panel = f"""## {status_emoji} GPU状态: {status_text}

### 📊 GPU概览
{gpu_summary}
### 👥 当前GPU用户
{user_display}

---
**状态说明**:
- 🟢 空闲: GPU可用
- 🟡 占用中: 有进程使用，但都是hy-worldplay用户
- 🔴 忙碌: 有非hy-worldplay用户占用大量显存(>500MB)
"""
    return panel, is_busy


def get_gpu_load_for_progress():
    """获取GPU负载用于进度条显示"""
    gpu_info = get_gpu_info()
    if not gpu_info:
        return 0, "无法获取GPU信息"
    
    # 计算平均GPU利用率
    total_util = sum(gpu['gpu_util'] for gpu in gpu_info) / len(gpu_info)
    
    # 获取显存使用情况
    mem_info = []
    for gpu in gpu_info:
        mem_info.append(f"GPU{gpu['id']}: {gpu['gpu_util']}%")
    
    return total_util, " | ".join(mem_info)


# GPU历史数据存储
_gpu_history = {
    "timestamps": [],
    "gpu_utils": [],  # 每个GPU的利用率历史
    "mem_used": [],   # 每个GPU的显存使用历史
    "max_history": 8,  # 保留最近8个数据点
    "last_update": 0,   # 上次更新时间
    "sample_interval": 5  # 采样间隔（秒）
}


def update_gpu_history():
    """更新GPU历史数据"""
    global _gpu_history
    gpu_info = get_gpu_info()
    if not gpu_info:
        return
    
    current_time = time.time()
    
    # 检查是否达到采样间隔
    if current_time - _gpu_history["last_update"] < _gpu_history["sample_interval"]:
        return
    
    _gpu_history["last_update"] = current_time
    _gpu_history["timestamps"].append(current_time)
    
    # 记录每个GPU的数据
    gpu_utils = [gpu['gpu_util'] for gpu in gpu_info]
    mem_used = [float(gpu['mem_used']) for gpu in gpu_info]
    
    _gpu_history["gpu_utils"].append(gpu_utils)
    _gpu_history["mem_used"].append(mem_used)
    
    # 限制历史数据长度
    if len(_gpu_history["timestamps"]) > _gpu_history["max_history"]:
        _gpu_history["timestamps"] = _gpu_history["timestamps"][-_gpu_history["max_history"]:]
        _gpu_history["gpu_utils"] = _gpu_history["gpu_utils"][-_gpu_history["max_history"]:]
        _gpu_history["mem_used"] = _gpu_history["mem_used"][-_gpu_history["max_history"]:]


def get_gpu_util_chart_data():
    """获取GPU利用率折线图数据"""
    import pandas as pd
    global _gpu_history
    if not _gpu_history["timestamps"] or not _gpu_history["gpu_utils"]:
        return pd.DataFrame(columns=["时间", "GPU", "利用率(%)"])
    
    data = []
    for i, (timestamp, utils) in enumerate(zip(_gpu_history["timestamps"], _gpu_history["gpu_utils"])):
        time_str = datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")
        for gpu_id, util in enumerate(utils):
            data.append({
                "时间": time_str,
                "GPU": f"GPU {gpu_id}",
                "利用率(%)": util
            })
    
    return pd.DataFrame(data)


def get_gpu_memory_chart_data():
    """获取GPU显存折线图数据"""
    import pandas as pd
    global _gpu_history
    if not _gpu_history["timestamps"] or not _gpu_history["mem_used"]:
        return pd.DataFrame(columns=["时间", "GPU", "显存(GB)"])
    
    data = []
    for i, (timestamp, mems) in enumerate(zip(_gpu_history["timestamps"], _gpu_history["mem_used"])):
        time_str = datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")
        for gpu_id, mem in enumerate(mems):
            data.append({
                "时间": time_str,
                "GPU": f"GPU {gpu_id}",
                "显存(GB)": mem
            })
    
    return pd.DataFrame(data)


def get_gpu_users_text():
    """获取GPU占用用户文本"""
    gpu_info = get_gpu_info()
    if not gpu_info:
        return "无法获取GPU信息"
    
    # 收集所有用户和其占用情况
    user_procs = {}
    for gpu in gpu_info:
        for proc in gpu.get('processes', []):
            user = proc['user']
            mem_mb = proc['mem_used_mb']
            pid = proc['pid']
            if user not in user_procs:
                user_procs[user] = []
            user_procs[user].append(f"GPU{gpu['id']}: {mem_mb:.0f}MB (PID:{pid})")
    
    if not user_procs:
        return "无占用用户"
    
    lines = []
    for user, procs in sorted(user_procs.items()):
        lines.append(f"• {user}")
        for p in procs:
            lines.append(f"  {p}")
    
    return "\n".join(lines)