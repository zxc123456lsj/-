from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.models import KnowledgeBase
from app.schemas.schemas import (
    RetrieveRequest, RetrieveResponse, RetrievedTextItem, RetrievedImageItem,
)
from app.services.retriever import Retriever

router = APIRouter(prefix="/api/v1", tags=["检索"])


@router.post("/retrieve", response_model=RetrieveResponse)
def retrieve(req: RetrieveRequest, db: Session = Depends(get_db)):
    kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == req.kb_id).first()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    text_results, image_results = Retriever.retrieve(
        query=req.query,
        kb_id=req.kb_id,
        top_k_text=req.top_k_text,
        top_k_image=req.top_k_image,
    )

    return RetrieveResponse(
        query=req.query,
        kb_id=req.kb_id,
        text_results=[RetrievedTextItem(**r) for r in text_results],
        image_results=[RetrievedImageItem(**r) for r in image_results],
    )
