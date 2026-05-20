from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.models import KnowledgeBase
from app.schemas.schemas import ChatRequest, ChatResponse, SourceInfo
from app.services.qa_service import QAService

router = APIRouter(prefix="/api/v1", tags=["问答"])


def get_qa_service():
    return QAService()


@router.post("/chat", response_model=ChatResponse)
def chat(
    req: ChatRequest,
    db: Session = Depends(get_db),
    qa: QAService = Depends(get_qa_service),
):
    kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == req.kb_id).first()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    answer, sources = qa.answer(query=req.query, kb_id=req.kb_id)

    return ChatResponse(
        query=req.query,
        answer=answer,
        sources=[SourceInfo(**s) for s in sources],
    )
