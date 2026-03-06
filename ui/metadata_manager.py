"""
HY-WorldPlay 元数据管理器

支持多进程安全的文件锁机制，确保守护进程和 Gradio UI 可以安全地共享状态。
"""

import json
import os
import shutil
import threading
import time
import uuid
import fcntl
from typing import Dict, List, Optional

from .config import IMAGE_DIR, METADATA_FILE
from .models import SavedImage, SavedPose, SavedItem


class FileLock:
    """
    多进程文件锁
    用于确保多个进程（Gradio UI 和守护进程）之间对元数据的安全访问
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
            # 确保锁文件目录存在
            os.makedirs(os.path.dirname(self.lock_file), exist_ok=True)
            self.fd = open(self.lock_file, 'w')
            if blocking:
                fcntl.flock(self.fd.fileno(), fcntl.LOCK_EX)
            else:
                fcntl.flock(self.fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
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
    
    def __enter__(self):
        self.acquire()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False


class MetadataManager:
    """元数据管理器 - 支持多进程安全访问"""
    
    _instance = None
    _lock = threading.Lock()
    
    # 多进程文件锁路径
    FILE_LOCK_PATH = os.path.join(os.path.dirname(METADATA_FILE), ".metadata.lock")
    
    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return
        self._initialized = True
        self._thread_lock = threading.Lock()  # 线程锁
        self._file_lock = FileLock(self.FILE_LOCK_PATH)  # 进程锁
        self.images: Dict[str, SavedImage] = {}
        self.poses: Dict[str, SavedPose] = {}
        self.items: Dict[str, SavedItem] = {}
        self._load()
    
    def _acquire_lock(self):
        """获取线程锁和文件锁"""
        self._thread_lock.acquire()
        self._file_lock.acquire()
    
    def _release_lock(self):
        """释放线程锁和文件锁"""
        self._file_lock.release()
        self._thread_lock.release()
    
    def _reload(self):
        """
        重新加载元数据（在持有锁的情况下调用）
        用于在多进程环境下获取最新状态
        """
        self.images.clear()
        self.poses.clear()
        self.items.clear()
        self._load_unsafe()
    
    def _load(self):
        """加载元数据（公共方法，会自动获取锁）"""
        with self._thread_lock:
            with self._file_lock:
                self._load_unsafe()
    
    def _load_unsafe(self):
        """加载元数据（不获取锁，调用者必须已持有锁）"""
        if os.path.exists(METADATA_FILE):
            try:
                with open(METADATA_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for img_data in data.get("images", []):
                        img = SavedImage.from_dict(img_data)
                        self.images[img.image_id] = img
                    for pose_data in data.get("poses", []):
                        pose = SavedPose.from_dict(pose_data)
                        self.poses[pose.pose_id] = pose
                    for item_data in data.get("items", []):
                        item = SavedItem.from_dict(item_data)
                        self.items[item.item_id] = item
            except Exception as e:
                print(f"加载元数据失败: {e}")
    
    def _save(self):
        """保存元数据（不获取锁，调用者必须已持有锁）"""
        try:
            data = {
                "images": [img.to_dict() for img in self.images.values()],
                "poses": [pose.to_dict() for pose in self.poses.values()],
                "items": [item.to_dict() for item in self.items.values()],
                "last_update": time.time()
            }
            # 使用临时文件写入，然后原子重命名，确保数据完整性
            temp_file = METADATA_FILE + ".tmp"
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(temp_file, METADATA_FILE)
        except Exception as e:
            print(f"保存元数据失败: {e}")
    
    # ========== 图片管理 ==========
    def add_image(self, image_path: str, prompt: str, name: str = "") -> SavedImage:
        self._acquire_lock()
        try:
            # 重新加载以获取最新状态
            self._reload()
            
            # 校验名称是否已存在（如果提供了名称）
            if name and name.strip():
                name = name.strip()
                for existing_img in self.images.values():
                    if existing_img.name == name:
                        raise ValueError(f"图片名称 '{name}' 已存在，请使用其他名称")
            
            image_id = str(uuid.uuid4())[:8]
            timestamp = time.time()
            
            ext = os.path.splitext(image_path)[1].lower() or ".jpg"
            saved_name = f"{image_id}{ext}"
            saved_path = os.path.join(IMAGE_DIR, saved_name)
            shutil.copy2(image_path, saved_path)
            
            img = SavedImage(
                image_id=image_id,
                image_path=saved_path,
                prompt=prompt,
                created_at=timestamp,
                name=name,
            )
            self.images[image_id] = img
            self._save()
            return img
        finally:
            self._release_lock()
    
    def get_image(self, image_id: str) -> Optional[SavedImage]:
        self._acquire_lock()
        try:
            self._reload()
            return self.images.get(image_id)
        finally:
            self._release_lock()
    
    def get_all_images(self) -> List[SavedImage]:
        self._acquire_lock()
        try:
            self._reload()
            return sorted(self.images.values(), key=lambda x: x.created_at, reverse=True)
        finally:
            self._release_lock()
    
    def delete_image(self, image_id: str) -> bool:
        self._acquire_lock()
        try:
            self._reload()
            if image_id in self.images:
                img = self.images[image_id]
                if os.path.exists(img.image_path):
                    os.remove(img.image_path)
                del self.images[image_id]
                self._save()
                return True
            return False
        finally:
            self._release_lock()
    
    # ========== 视角管理 ==========
    def add_pose(self, name: str, pose_code: str, num_frames: int) -> SavedPose:
        self._acquire_lock()
        try:
            self._reload()
            
            pose_id = str(uuid.uuid4())[:8]
            timestamp = time.time()
            
            pose = SavedPose(
                pose_id=pose_id,
                name=name,
                pose_code=pose_code,
                num_frames=num_frames,
                created_at=timestamp,
            )
            self.poses[pose_id] = pose
            self._save()
            return pose
        finally:
            self._release_lock()
    
    def get_pose(self, pose_id: str) -> Optional[SavedPose]:
        self._acquire_lock()
        try:
            self._reload()
            return self.poses.get(pose_id)
        finally:
            self._release_lock()
    
    def get_all_poses(self) -> List[SavedPose]:
        self._acquire_lock()
        try:
            self._reload()
            return sorted(self.poses.values(), key=lambda x: x.created_at, reverse=True)
        finally:
            self._release_lock()
    
    def delete_pose(self, pose_id: str) -> bool:
        self._acquire_lock()
        try:
            self._reload()
            if pose_id in self.poses:
                del self.poses[pose_id]
                self._save()
                return True
            return False
        finally:
            self._release_lock()
    
    # ========== 项目（视频任务）管理 ==========
    def add_item(self, image_id: str, pose_id: str, model_type: str, seed: int = 1) -> Optional[SavedItem]:
        """
        添加视频生成任务
        注意：创建时状态为 pending（待处理），不是 processing
        
        Args:
            image_id: 图片ID
            pose_id: 视角ID
            model_type: 模型类型
            seed: 随机种子，默认值为1
        """
        self._acquire_lock()
        try:
            self._reload()
            
            image = self.images.get(image_id)
            pose = self.poses.get(pose_id)
            if not image or not pose:
                return None
            
            item_id = str(uuid.uuid4())[:8]
            timestamp = time.time()
            
            item = SavedItem(
                item_id=item_id,
                image_id=image_id,
                image_path=image.image_path,
                prompt=image.prompt,
                pose_id=pose_id,
                pose_code=pose.pose_code,
                num_frames=pose.num_frames,
                model_type=model_type,
                created_at=timestamp,
                status="pending",  # 初始状态为 pending，等待守护进程处理
                seed=seed,
            )
            self.items[item_id] = item
            self._save()
            return item
        finally:
            self._release_lock()
    
    def get_item(self, item_id: str) -> Optional[SavedItem]:
        self._acquire_lock()
        try:
            self._reload()
            return self.items.get(item_id)
        finally:
            self._release_lock()
    
    def get_all_items(self) -> List[SavedItem]:
        self._acquire_lock()
        try:
            self._reload()
            return sorted(self.items.values(), key=lambda x: x.created_at, reverse=True)
        finally:
            self._release_lock()
    
    def update_item(self, item_id: str, **kwargs):
        """
        更新任务状态
        
        Args:
            item_id: 任务ID
            **kwargs: 要更新的字段（status, video_path, error_msg 等）
        """
        self._acquire_lock()
        try:
            self._reload()
            if item_id in self.items:
                item = self.items[item_id]
                for key, value in kwargs.items():
                    if hasattr(item, key):
                        setattr(item, key, value)
                self._save()
                return True
            return False
        finally:
            self._release_lock()
    
    def delete_item(self, item_id: str) -> bool:
        self._acquire_lock()
        try:
            self._reload()
            if item_id in self.items:
                item = self.items[item_id]
                if item.video_path and os.path.exists(item.video_path):
                    os.remove(item.video_path)
                del self.items[item_id]
                self._save()
                return True
            return False
        finally:
            self._release_lock()
    
    def get_pending_items(self) -> List[SavedItem]:
        """获取所有待处理的任务（供守护进程使用）"""
        self._acquire_lock()
        try:
            self._reload()
            return [item for item in self.items.values() if item.status == "pending"]
        finally:
            self._release_lock()
    
    def reset_stuck_tasks(self) -> int:
        """
        重置卡住的任务为 pending 状态
        将非 pending 且非 completed 的任务重置为 pending
        
        Returns:
            重置的任务数量
        """
        self._acquire_lock()
        try:
            self._reload()
            reset_count = 0
            for item in self.items.values():
                if item.status not in ("pending", "completed"):
                    item.status = "pending"
                    item.error_msg = ""  # 清空错误信息
                    reset_count += 1
            if reset_count > 0:
                self._save()
            return reset_count
        finally:
            self._release_lock()
    
    def get_failed_items(self) -> List[SavedItem]:
        """获取所有失败的任务"""
        self._acquire_lock()
        try:
            self._reload()
            return [item for item in self.items.values() if item.status == "failed"]
        finally:
            self._release_lock()


# 全局单例
metadata_manager = MetadataManager()
