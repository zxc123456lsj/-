import os
import uuid
import json
from fastapi import APIRouter, UploadFile, File, HTTPException
from kafka import KafkaProducer
from orm_model import Session, File

router = APIRouter()

producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

@router.post("/upload/document")
async def upload_document(file: UploadFile = File(...)):
    # 1. 保存原始文件
    os.makedirs("uploads", exist_ok=True)
    ext = os.path.splitext(file.filename)[1]
    save_name = str(uuid.uuid4()) + ext
    save_path = os.path.join("uploads", save_name)
    with open(save_path, "wb") as f:
        f.write(await file.read())

    # 2. 数据库记录
    with Session() as session:
        record = File(
            filename=file.filename,
            filepath=save_path,
            filestate="uploaded"
        )
        session.add(record)
        session.commit()
        file_id = record.id

    # 3. 发送 Kafka 异步解析
    producer.send("rag-data", value={
        "file_name": file.filename,
        "file_path": save_path,
        "id": file_id
    })
    producer.flush()

    return {"code": 0, "msg": "upload success, parsing started", "file_id": file_id}