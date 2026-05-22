"""
文件上传服务
处理文件上传、存储和消息发送
"""
import os
import uuid
import shutil
from kafka import KafkaProducer
import json
import config
from models.file_model import create_file_record, delete_file as delete_file_record
from utils.milvus_client import delete_by_file_id

# 全局 Kafka Producer（延迟加载）
_kafka_producer = None


def get_kafka_producer() -> KafkaProducer:
    """获取 Kafka Producer 单例"""
    global _kafka_producer
    if _kafka_producer is None:
        _kafka_producer = KafkaProducer(
            bootstrap_servers=config.KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
    return _kafka_producer


def save_uploaded_file(file_content: bytes, original_filename: str) -> dict:
    """
    保存上传的文件

    Args:
        file_content: 文件二进制内容
        original_filename: 原始文件名

    Returns:
        {
            "file_id": int,
            "save_path": str,
            "filename": str
        }
    """
    # 生成唯一文件名
    file_extension = os.path.splitext(original_filename)[1]
    unique_name = str(uuid.uuid4())
    save_filename = f"{unique_name}{file_extension}"
    save_path = os.path.join(config.UPLOAD_DIR, save_filename)

    # 保存文件
    with open(save_path, "wb") as f:
        f.write(file_content)

    # 创建数据库记录
    file_id = create_file_record(original_filename, save_path)

    return {
        "file_id": file_id,
        "save_path": save_path,
        "filename": original_filename
    }


def send_to_kafka(file_id: int, filename: str, file_path: str):
    """
    发送文档解析消息到 Kafka

    Args:
        file_id: 文件ID
        filename: 文件名
        file_path: 文件路径
    """
    producer = get_kafka_producer()

    message = {
        "file_name": filename,
        "file_path": file_path,
        "id": file_id
    }

    producer.send(config.KAFKA_TOPIC_DOCUMENT, value=message)
    producer.flush()
    print(f"Message sent to Kafka: {message}")


def handle_file_upload(file_content: bytes, original_filename: str) -> dict:
    """
    处理文件上传完整流程

    Args:
        file_content: 文件二进制内容
        original_filename: 原始文件名

    Returns:
        上传结果信息
    """
    # 1. 保存文件
    result = save_uploaded_file(file_content, original_filename)

    # 2. 发送 Kafka 消息
    send_to_kafka(
        file_id=result["file_id"],
        filename=result["filename"],
        file_path=result["save_path"]
    )

    return {
        "code": 0,
        "message": "success",
        "data": {
            "file_id": result["file_id"],
            "filename": result["filename"],
            "status": "已上传"
        }
    }


def delete_file_and_data(file_id: int) -> dict:
    """
    删除文件及相关数据

    Args:
        file_id: 文件ID

    Returns:
        删除结果
    """
    from models.file_model import get_file_by_id

    # 1. 获取文件信息
    file_info = get_file_by_id(file_id)
    if not file_info:
        return {
            "code": 404,
            "message": "file not found"
        }

    # 2. 删除本地文件
    try:
        if os.path.exists(file_info["filepath"]):
            os.remove(file_info["filepath"])
    except Exception as e:
        print(f"Error deleting file: {e}")

    # 3. 删除 Milvus 向量
    try:
        delete_by_file_id(file_id)
    except Exception as e:
        print(f"Error deleting vectors: {e}")

    # 4. 删除数据库记录
    delete_file_record(file_id)

    return {
        "code": 0,
        "message": "deleted"
    }


def list_all_files() -> dict:
    """
    获取所有文件列表

    Returns:
        文件列表
    """
    from models.file_model import list_files

    files = list_files()

    return {
        "code": 0,
        "data": files
    }
