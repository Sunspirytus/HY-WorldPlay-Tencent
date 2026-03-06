"""
HY-WorldPlay UI 工具函数
"""

import os
import re
import requests
from datetime import datetime

from .config import DESCRIBE_API_URL, VALID_ACTIONS


def format_datetime(timestamp: float) -> str:
    """格式化时间戳"""
    return datetime.fromtimestamp(timestamp).strftime("%m-%d %H:%M")


def generate_prompt_from_image(image_path: str, language: str = "zh") -> str:
    """调用API从图片生成prompt"""
    if not image_path or not os.path.exists(image_path):
        return "错误：请先上传图片"
    
    try:
        with open(image_path, "rb") as f:
            files = {"file": f}
            data = {"language": language}
            response = requests.post(DESCRIBE_API_URL, files=files, data=data, timeout=60)
            response.raise_for_status()
            result = response.json()
            if isinstance(result, dict):
                return result.get("description", result.get("prompt", str(result)))
            return str(result)
    except requests.exceptions.RequestException as e:
        return f"API请求失败: {str(e)}"
    except Exception as e:
        return f"错误: {str(e)}"


def calculate_pose_frames(pose_code: str) -> int:
    """计算轨迹代码的轨迹数量，公式: (帧数-1)/4"""
    if not pose_code or not pose_code.strip():
        return 0
    
    total_frames = 0
    actions = [a.strip() for a in pose_code.split(',')]
    for action in actions:
        if not action:
            continue
        match = re.match(r'^([a-z]+)-(\d+)$', action)
        if match:
            total_frames += int(match.group(2))
    return (total_frames - 1) // 4


def validate_pose_code(pose_code: str) -> tuple:
    """
    验证轨迹代码格式是否正确
    格式: `动作-持续时间`, 多个动作用逗号分隔
    - 移动: w(前), s(后), a(左), d(右)
    - 旋转: up(抬头), down(低头), left(左转), right(右转)
    
    返回: (is_valid: bool, error_msg: str)
    """
    if not pose_code or not pose_code.strip():
        return False, "轨迹代码不能为空"
    
    # 去除多余空格
    pose_code = pose_code.strip()
    
    # 分割多个动作
    actions = [a.strip() for a in pose_code.split(',')]
    
    for action in actions:
        if not action:
            continue
        
        # 匹配 动作-数字 格式
        match = re.match(r'^([a-z]+)-(\d+)$', action)
        if not match:
            return False, f"格式错误: '{action}' 不符合 '动作-持续时间' 格式"
        
        action_name = match.group(1)
        duration = match.group(2)
        
        if action_name not in VALID_ACTIONS:
            return False, f"无效动作: '{action_name}', 有效动作: {', '.join(sorted(VALID_ACTIONS))}"
        
        try:
            duration_val = int(duration)
            if duration_val <= 0:
                return False, f"持续时间必须大于0: '{action}'"
        except ValueError:
            return False, f"持续时间必须是数字: '{action}'"
    
    return True, "格式正确"