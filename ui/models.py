"""
HY-WorldPlay UI 数据模型
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class SavedImage:
    """保存的图片数据"""
    image_id: str
    image_path: str
    prompt: str
    created_at: float
    name: str = ""  # 图片名称
    
    def to_dict(self) -> dict:
        return {
            "image_id": self.image_id,
            "image_path": self.image_path,
            "prompt": self.prompt,
            "created_at": self.created_at,
            "name": self.name,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "SavedImage":
        # 兼容旧数据，如果没有name字段则默认为空字符串
        if "name" not in data:
            data["name"] = ""
        return cls(**data)


@dataclass
class SavedPose:
    """保存的视角轨迹数据"""
    pose_id: str
    name: str
    pose_code: str
    num_frames: int
    created_at: float
    
    def to_dict(self) -> dict:
        return {
            "pose_id": self.pose_id,
            "name": self.name,
            "pose_code": self.pose_code,
            "num_frames": self.num_frames,
            "created_at": self.created_at,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "SavedPose":
        return cls(**data)


@dataclass
class SavedItem:
    """保存的完整项目（视频生成任务）"""
    item_id: str
    image_id: str
    image_path: str
    prompt: str
    pose_id: str
    pose_code: str
    num_frames: int
    model_type: str
    created_at: float
    video_path: Optional[str] = None
    status: str = "pending"  # pending, processing, completed, failed
    error_msg: str = ""
    seed: int = 1  # 随机种子，默认值为1
    
    def to_dict(self) -> dict:
        return {
            "item_id": self.item_id,
            "image_id": self.image_id,
            "image_path": self.image_path,
            "prompt": self.prompt,
            "pose_id": self.pose_id,
            "pose_code": self.pose_code,
            "num_frames": self.num_frames,
            "model_type": self.model_type,
            "created_at": self.created_at,
            "video_path": self.video_path,
            "status": self.status,
            "error_msg": self.error_msg,
            "seed": self.seed,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "SavedItem":
        defaults = {"video_path": None, "status": "pending", "error_msg": "", "seed": 1}
        for key, default in defaults.items():
            if key not in data:
                data[key] = default
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
