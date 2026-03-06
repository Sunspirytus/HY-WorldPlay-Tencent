"""
HY-WorldPlay UI 配置文件
"""

import os

# 获取项目根目录（ui的父目录）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ==================== 数据存储路径 ====================
DATA_DIR = "data/gradio_storage"
IMAGE_DIR = os.path.join(PROJECT_ROOT, DATA_DIR, "images")
POSE_DIR = os.path.join(PROJECT_ROOT, DATA_DIR, "poses")
VIDEO_DIR = os.path.join(PROJECT_ROOT, DATA_DIR, "videos")
METADATA_FILE = os.path.join(PROJECT_ROOT, DATA_DIR, "metadata.json")

# 确保目录存在
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(POSE_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)

# ==================== API配置 ====================
DESCRIBE_API_URL = "http://172.18.4.63:28000/describe/file"

# ==================== 视角动作预设 ====================
POSE_PRESETS = {
    "向前移动": "w-31",
    "向后移动": "s-31",
    "向左移动": "a-31",
    "向右移动": "d-31",
    "向上看": "up-31",
    "向下看": "down-31",
    "向左转": "left-31",
    "向右转": "right-31",
    "前进+右转": "w-15, right-15",
    "前进+向上": "w-15, up-15",
    "复杂轨迹1": "w-10, right-5, w-10, left-5",
    "复杂轨迹2": "w-5, up-5, w-5, down-5, w-10",
}

# ==================== 有效动作列表 ====================
VALID_ACTIONS = {'w', 's', 'a', 'd', 'up', 'down', 'left', 'right'}