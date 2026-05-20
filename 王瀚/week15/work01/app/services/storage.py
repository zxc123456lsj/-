import os
import shutil
import hashlib
from pathlib import Path
from typing import BinaryIO

from app.config import settings


class StorageService:
    @staticmethod
    def save_pdf(file: BinaryIO, filename: str) -> tuple[str, str, int]:
        file_bytes = file.read()
        file_hash = hashlib.md5(file_bytes).hexdigest()
        safe_name = f"{file_hash}_{filename}"
        save_path = os.path.join(settings.pdf_dir, safe_name)
        file_size = len(file_bytes)
        with open(save_path, "wb") as f:
            f.write(file_bytes)
        return save_path, safe_name, file_size

    @staticmethod
    def save_image(image_bytes: bytes, doc_id: str, page_num: int, image_index: int) -> str:
        ext = ".png"
        image_name = f"{doc_id}_p{page_num}_i{image_index}{ext}"
        save_path = os.path.join(settings.image_dir, image_name)
        with open(save_path, "wb") as f:
            f.write(image_bytes)
        return save_path

    @staticmethod
    def save_parsed_content(doc_id: str, content: str) -> str:
        save_path = os.path.join(settings.parsed_dir, f"{doc_id}.md")
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(content)
        return save_path

    @staticmethod
    def delete_file(file_path: str) -> bool:
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                return True
            return False
        except Exception:
            return False

    @staticmethod
    def get_file_url(relative_path: str) -> str:
        filename = os.path.basename(relative_path)
        return f"/static/images/{filename}"
