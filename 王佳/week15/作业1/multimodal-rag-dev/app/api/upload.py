from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from sqlalchemy.orm import Session

from app.models.database import File as FileModel, ProcessingLog, get_db
from app.utils.config import MAX_UPLOAD_SIZE, UPLOAD_DIR
from app.utils.hash_utils import compute_file_hash
from app.utils.kafka_utils import get_producer, send_message

router = APIRouter(prefix="/api", tags=["files"])


@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    knowledge_base_id: int = Query(default=1),
):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="仅支持PDF文件")

    content = await file.read()
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail=f"文件过大，最大支持 {MAX_UPLOAD_SIZE // 1024 // 1024}MB")

    upload_dir = Path(UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / file.filename

    with open(file_path, "wb") as f:
        f.write(content)

    file_hash = compute_file_hash(file_path)

    db: Session = next(get_db())
    try:
        existing = db.query(FileModel).filter(FileModel.file_hash == file_hash).first()
        if existing:
            file_path.unlink()  # 删除重复文件
            return {
                "status": "success",
                "file_id": existing.id,
                "filename": existing.filename,
                "message": "文件已存在，跳过上传",
            }

        record = FileModel(
            filename=file.filename,
            filepath=str(file_path.resolve()),
            file_hash=file_hash,
            filestate="pending",
        )
        db.add(record)
        db.flush()

        log = ProcessingLog(file_id=record.id, status="pending")
        db.add(log)
        db.commit()

        producer = get_producer()
        message = {
            "file_id": record.id,
            "filename": file.filename,
            "filepath": str(file_path.resolve()),
            "file_hash": file_hash,
            "knowledge_base_id": knowledge_base_id,
        }
        send_message(producer, message)
        producer.close()

        return {
            "status": "success",
            "file_id": record.id,
            "filename": file.filename,
            "message": "文件已上传，正在后台解析",
        }
    finally:
        db.close()


@router.get("/files/{file_id}/status")
async def get_file_status(file_id: int):
    db: Session = next(get_db())
    try:
        record = db.query(FileModel).filter(FileModel.id == file_id).first()
        if not record:
            raise HTTPException(status_code=404, detail="文件不存在")

        return {
            "file_id": record.id,
            "filename": record.filename,
            "status": record.filestate,
            "created_at": record.created_at.isoformat() if record.created_at else None,
        }
    finally:
        db.close()


@router.get("/files")
async def list_files(
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=20, ge=1, le=100),
    status: str | None = Query(default=None),
):
    db: Session = next(get_db())
    try:
        query = db.query(FileModel)
        if status:
            query = query.filter(FileModel.filestate == status)

        total = query.count()
        files = (
            query
            .order_by(FileModel.created_at.desc())
            .offset((page - 1) * limit)
            .limit(limit)
            .all()
        )

        return {
            "total": total,
            "page": page,
            "limit": limit,
            "files": [
                {
                    "id": f.id,
                    "filename": f.filename,
                    "status": f.filestate,
                    "created_at": f.created_at.isoformat() if f.created_at else None,
                }
                for f in files
            ],
        }
    finally:
        db.close()
