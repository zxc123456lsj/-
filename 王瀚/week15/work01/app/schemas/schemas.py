from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class KnowledgeBaseCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = ""


class KnowledgeBaseResponse(BaseModel):
    id: str
    name: str
    description: str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class KnowledgeBaseList(BaseModel):
    total: int
    items: list[KnowledgeBaseResponse]


class DocumentResponse(BaseModel):
    id: str
    kb_id: str
    filename: str
    file_size: int
    page_count: int
    parse_status: str
    error_message: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class DocumentList(BaseModel):
    total: int
    items: list[DocumentResponse]


class DocumentUploadResponse(BaseModel):
    id: str
    filename: str
    parse_status: str
    message: str


class ParseStatusResponse(BaseModel):
    doc_id: str
    filename: str
    parse_status: str
    page_count: int
    error_message: Optional[str] = None


class RetrieveRequest(BaseModel):
    kb_id: str = Field(..., description="知识库ID")
    query: str = Field(..., min_length=1, description="查询文本")
    top_k_text: int = Field(default=5, ge=1, le=20)
    top_k_image: int = Field(default=3, ge=1, le=20)


class RetrievedTextItem(BaseModel):
    chunk_id: str
    doc_id: str
    content: str
    page_num: int
    score: float


class RetrievedImageItem(BaseModel):
    image_id: str
    doc_id: str
    file_path: str
    page_num: int
    caption: Optional[str] = None
    score: float


class RetrieveResponse(BaseModel):
    query: str
    kb_id: str
    text_results: list[RetrievedTextItem]
    image_results: list[RetrievedImageItem]


class ChatRequest(BaseModel):
    kb_id: str = Field(..., description="知识库ID")
    query: str = Field(..., min_length=1, description="用户问题")


class SourceInfo(BaseModel):
    doc_id: str
    filename: str
    page_num: int
    chunk_id: Optional[str] = None
    image_id: Optional[str] = None


class ChatResponse(BaseModel):
    query: str
    answer: str
    sources: list[SourceInfo]
