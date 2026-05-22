from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.models import KnowledgeBase
from app.schemas.schemas import (
    KnowledgeBaseCreate, KnowledgeBaseResponse, KnowledgeBaseList,
)

router = APIRouter(prefix="/api/v1/knowledge-bases", tags=["知识库管理"])


@router.post("", response_model=KnowledgeBaseResponse, status_code=201)
def create_kb(body: KnowledgeBaseCreate, db: Session = Depends(get_db)):
    kb = KnowledgeBase(name=body.name, description=body.description or "")
    db.add(kb)
    db.commit()
    db.refresh(kb)
    return kb


@router.get("", response_model=KnowledgeBaseList)
def list_kbs(db: Session = Depends(get_db)):
    items = db.query(KnowledgeBase).order_by(KnowledgeBase.created_at.desc()).all()
    return KnowledgeBaseList(total=len(items), items=items)


@router.get("/{kb_id}", response_model=KnowledgeBaseResponse)
def get_kb(kb_id: str, db: Session = Depends(get_db)):
    kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    return kb


@router.delete("/{kb_id}", status_code=204)
def delete_kb(kb_id: str, db: Session = Depends(get_db)):
    kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    db.delete(kb)
    db.commit()
