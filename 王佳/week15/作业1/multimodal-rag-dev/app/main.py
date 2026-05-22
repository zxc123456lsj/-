import os

from dotenv import load_dotenv
from fastapi import FastAPI

from app.api.chat import router as chat_router
from app.api.upload import router as upload_router
from app.models.database import init_db
from app.utils.kafka_utils import get_producer

load_dotenv()

app = FastAPI(
    title="多模态RAG问答系统",
    description="支持PDF文档解析、多模态向量检索、智能问答",
    version="1.0.0",
)

app.include_router(upload_router)
app.include_router(chat_router)


@app.on_event("startup")
def startup():
    init_db()


@app.get("/api/health")
def health_check():
    services = {"kafka": "unknown", "milvus": "unknown", "mineru": "unknown"}
    try:
        producer = get_producer()
        producer.close()
        services["kafka"] = "connected"
    except Exception:
        services["kafka"] = "disconnected"

    try:
        from pymilvus import connections
        connections.connect("default", uri=os.getenv("MILVUS_URI", "http://localhost:19530"), timeout=3)
        services["milvus"] = "connected"
        connections.disconnect("default")
    except Exception:
        services["milvus"] = "disconnected"

    try:
        import httpx
        r = httpx.get(f"{os.getenv('MINERU_API_URL', 'http://localhost:30000')}", timeout=3)
        services["mineru"] = "available" if r.status_code < 500 else "unavailable"
    except Exception:
        services["mineru"] = "unavailable"

    return {"status": "healthy" if all(v in ("connected", "available") for v in services.values()) else "degraded", "services": services}
