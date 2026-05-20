import json
import os
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session

from app.config import settings
from app.core.database import get_db
from app.core.models import KnowledgeBase, Document, ParseStatus
from app.schemas.schemas import (
    DocumentResponse, DocumentList, DocumentUploadResponse, ParseStatusResponse,
)
from app.services.storage import StorageService

router = APIRouter(prefix="/api/v1", tags=["文档管理"])

ALLOWED_EXTENSIONS = {".pdf"}


@router.post("/knowledge-bases/{kb_id}/documents",
             response_model=DocumentUploadResponse, status_code=201)
def upload_document(
    kb_id: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    file_path, safe_name, file_size = StorageService.save_pdf(file.file, file.filename or "upload.pdf")

    doc = Document(
        kb_id=kb_id,
        filename=safe_name,
        file_path=file_path,
        file_size=file_size,
        parse_status=ParseStatus.pending,
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    _send_parse_message(doc.id, file_path)

    return DocumentUploadResponse(
        id=doc.id,
        filename=safe_name,
        parse_status=doc.parse_status.value,
        message="Document uploaded, parsing queued",
    )


@router.get("/knowledge-bases/{kb_id}/documents", response_model=DocumentList)
def list_documents(kb_id: str, db: Session = Depends(get_db)):
    kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    items = db.query(Document).filter(
        Document.kb_id == kb_id
    ).order_by(Document.created_at.desc()).all()
    return DocumentList(total=len(items), items=items)


@router.delete("/knowledge-bases/{kb_id}/documents/{doc_id}", status_code=204)
def delete_document(kb_id: str, doc_id: str, db: Session = Depends(get_db)):
    doc = db.query(Document).filter(
        Document.id == doc_id, Document.kb_id == kb_id
    ).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    StorageService.delete_file(doc.file_path)
    db.delete(doc)
    db.commit()
    from app.services.vector_store import VectorStore
    VectorStore.delete_by_doc(doc_id)


@router.get("/documents/{doc_id}/parse-status", response_model=ParseStatusResponse)
def get_parse_status(doc_id: str, db: Session = Depends(get_db)):
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return ParseStatusResponse(
        doc_id=doc.id,
        filename=doc.filename,
        parse_status=doc.parse_status.value,
        page_count=doc.page_count,
        error_message=doc.error_message,
    )


def _send_parse_message(doc_id: str, file_path: str):
    try:
        from confluent_kafka import Producer
        producer = Producer({"bootstrap.servers": settings.kafka_bootstrap_servers})
        message = json.dumps({"doc_id": doc_id, "file_path": file_path})
        producer.produce(settings.kafka_parse_topic, value=message.encode("utf-8"))
        producer.flush(timeout=5)
    except ImportError:
        pass
    except Exception:
        pass
