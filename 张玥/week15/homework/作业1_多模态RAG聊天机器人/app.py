"""多模态 RAG 聊天机器人作业版 FastAPI 接口。"""

from __future__ import annotations

from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel

from rag_core import (
    build_index,
    chat_with_rag,
    get_document,
    init_storage,
    parse_document_task,
    render_markdown_with_images,
    upload_document_bytes,
)


app = FastAPI(title="Week15 多模态 RAG 聊天机器人作业版")


class ChatRequest(BaseModel):
    question: str
    top_k: int = 3


@app.on_event("startup")
def startup() -> None:
    init_storage()


@app.post("/upload/document")
async def upload_document_api(file: UploadFile = File(...)) -> dict:
    content = await file.read()
    record = upload_document_bytes(file.filename or "uploaded.txt", content)
    return {
        "document_id": record.id,
        "filename": record.filename,
        "status": record.status,
    }


@app.post("/worker/parse/{document_id}")
def parse_document_api(document_id: int, use_mineru: bool = False) -> dict:
    record = parse_document_task(document_id, use_mineru=use_mineru)
    return {
        "document_id": record.id,
        "filename": record.filename,
        "status": record.status,
        "parsed_path": record.parsed_path,
    }


@app.post("/index/build")
def build_index_api() -> dict:
    index = build_index()
    return {
        "chunk_count": len(index["chunks"]),
        "status": "ok",
    }


@app.post("/chat")
def chat_api(request: ChatRequest) -> dict:
    result = chat_with_rag(request.question, top_k=request.top_k)
    return {
        "question": request.question,
        "answer": result["answer"],
        "hits": result["hits"],
        "render_parts": render_markdown_with_images(result["answer"]),
    }


@app.get("/documents/{document_id}")
def get_document_api(document_id: int) -> dict:
    record = get_document(document_id)
    return {
        "document_id": record.id,
        "filename": record.filename,
        "filepath": record.filepath,
        "status": record.status,
        "parsed_path": record.parsed_path,
    }
