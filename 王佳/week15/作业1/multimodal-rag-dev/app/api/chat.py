from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.services.chat import generate_answer

router = APIRouter(prefix="/api", tags=["chat"])


class ChatRequest(BaseModel):
    query: str = Field(..., description="用户问题")
    knowledge_base_ids: list[int] = Field(default=[1], description="知识库ID列表")
    top_k: int = Field(default=5, ge=1, le=20, description="返回片段数量")


@router.post("/chat")
def chat(req: ChatRequest):
    return generate_answer(req.query, top_k=req.top_k)
