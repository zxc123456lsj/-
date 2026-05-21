import os
import re
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from app.models.database import File as FileModel, ProcessingLog
from app.models.database import get_db
from app.services.milvus_utils import connect_milvus, disconnect_milvus, insert_data
from app.utils.kafka_utils import get_consumer

load_dotenv()

MINERU_API_URL = os.getenv("MINERU_API_URL", "http://localhost:30000")
BGE_MODEL_PATH = os.getenv("BGE_MODEL_PATH", "BAAI/bge-small-zh-v1.5")
CLIP_MODEL_PATH = os.getenv("CLIP_MODEL_PATH", "jinaai/jina-clip-v2")

CHUNK_SIZE = 256

bge_model: SentenceTransformer | None = None
clip_model: SentenceTransformer | None = None


def load_models() -> tuple[SentenceTransformer, SentenceTransformer]:
    global bge_model, clip_model
    if bge_model is None:
        bge_model = SentenceTransformer(BGE_MODEL_PATH)
    if clip_model is None:
        clip_model = SentenceTransformer(CLIP_MODEL_PATH, trust_remote_code=True)
    return bge_model, clip_model


def parse_pdf_with_mineru(filepath: str) -> dict[str, Any]:
    """调用 MinerU API 解析 PDF，返回 markdown 和图片列表"""
    with open(filepath, "rb") as f:
        files = {"file": f}
        response = httpx.post(
            f"{MINERU_API_URL}/api/v1/parse",
            files=files,
            timeout=300,  # 5分钟超时
        )
        response.raise_for_status()
        return response.json()


def split_text_into_chunks(text: str, chunk_size: int = CHUNK_SIZE) -> list[str]:
    """将文本按句子边界分割为指定大小的chunk"""
    sentences = re.split(r"(?<=[。！？.!?\n])\s*", text)
    chunks: list[str] = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    return chunks


def process_message(message: dict[str, Any]) -> None:
    file_id = message["file_id"]
    filename = message["filename"]
    filepath = message["filepath"]

    db = next(get_db())
    try:
        file_record = db.query(FileModel).filter(FileModel.id == file_id).first()
        if not file_record:
            return
        file_record.filestate = "processing"
        db.add(ProcessingLog(file_id=file_id, status="processing"))
        db.commit()

        # 1. 调用 MinerU 解析
        parse_result = parse_pdf_with_mineru(filepath)
        markdown_content = parse_result.get("markdown", "")
        images = parse_result.get("images", [])

        # 2. 加载模型
        bge, clip = load_models()

        # 3. 分割chunk并向量化
        chunks = split_text_into_chunks(markdown_content)
        if not chunks:
            chunks = [markdown_content[:CHUNK_SIZE]]

        connect_milvus()

        # 4. BGE文本向量编码 + 插入
        data_batch: list[dict] = []
        for idx, chunk in enumerate(chunks):
            text_vector = bge.encode(chunk, normalize_embeddings=True).tolist()
            clip_text_vector = clip.encode(chunk, normalize_embeddings=True).tolist()
            # CLIP图像向量用零向量占位（无对应图像时）
            data_batch.append({
                "text_vector": text_vector,
                "clip_text_vector": clip_text_vector,
                "clip_image_vector": [0.0] * 1024,
                "text": chunk,
                "db_id": file_id,
                "file_name": filename,
                "file_path": filepath,
                "chunk_index": idx,
            })

            # 每100条批量插入
            if len(data_batch) >= 100:
                insert_data(data_batch)
                data_batch = []

        # 5. CLIP编码图像并插入
        for img_idx, image_path in enumerate(images):
            if Path(image_path).exists():
                try:
                    from PIL import Image
                    img = Image.open(image_path)
                    image_vector = clip.encode(img, normalize_embeddings=True).tolist()
                    data_batch.append({
                        "text_vector": [0.0] * 512,
                        "clip_text_vector": [0.0] * 1024,
                        "clip_image_vector": image_vector,
                        "text": f"[IMAGE] {filename} 第{img_idx + 1}张图",
                        "db_id": file_id,
                        "file_name": filename,
                        "file_path": image_path,
                        "chunk_index": img_idx,
                    })
                except Exception:
                    pass

        if data_batch:
            insert_data(data_batch)

        disconnect_milvus()

        file_record.filestate = "completed"
        db.add(ProcessingLog(file_id=file_id, status="completed"))
        db.commit()

    except Exception as e:
        file_record.filestate = "failed"
        db.add(ProcessingLog(file_id=file_id, status="failed", error_message=str(e)))
        db.commit()
        raise
    finally:
        db.close()


def main() -> None:
    print("[Worker] 启动离线解析Worker...")
    load_models()  # 预热模型
    consumer = get_consumer()
    print(f"[Worker] 开始监听 Kafka topic，等待消息...")
    for msg in consumer:
        try:
            print(f"[Worker] 收到消息: file_id={msg.value.get('file_id')}")
            process_message(msg.value)
            consumer.commit()
            print(f"[Worker] 处理完成: file_id={msg.value.get('file_id')}")
        except Exception as e:
            print(f"[Worker] 处理失败: {e}")
            # 不commit，让消息重试


if __name__ == "__main__":
    main()
