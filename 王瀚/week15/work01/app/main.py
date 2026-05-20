import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.core.database import init_db
from app.api import knowledge_base, document, retrieve, chat


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="MultiModal RAG System",
    description="多模态检索增强生成问答系统 - 支持图文混排PDF知识库的智能问答",
    version="0.1.0",
    lifespan=lifespan,
)

static_dir = os.path.join(settings.storage_root, "images")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static/images", StaticFiles(directory=static_dir), name="images")

app.include_router(knowledge_base.router)
app.include_router(document.router)
app.include_router(retrieve.router)
app.include_router(chat.router)


@app.get("/health")
def health():
    return {"status": "ok", "service": "mrag"}
