import os
import sys
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.main import app
from app.core.database import init_db, engine
from app.core.models import Base


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True, scope="session")
def setup_db():
    Base.metadata.create_all(bind=engine)
    init_db()
    yield


@pytest.fixture
def auto_mocks():
    with patch("app.services.embedding_service.EmbeddingService.embed_text_bge",
               return_value=[0.0] * 1024), \
         patch("app.services.embedding_service.EmbeddingService.embed_text_clip",
               return_value=[0.0] * 512), \
         patch("app.services.embedding_service.EmbeddingService.embed_image_clip",
               return_value=[0.0] * 512):
        yield


@pytest.fixture
def sample_kb(client):
    resp = client.post("/api/v1/knowledge-bases", json={
        "name": "测试知识库",
        "description": "用于单元测试的知识库",
    })
    assert resp.status_code == 201
    return resp.json()


@pytest.fixture
def sample_pdf_path():
    pdf_dir = os.path.join(os.path.dirname(__file__), "..", "storage", "pdfs")
    os.makedirs(pdf_dir, exist_ok=True)

    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    pdf_path = os.path.join(pdf_dir, "test_sample.pdf")
    c = canvas.Canvas(pdf_path, pagesize=A4)
    c.drawString(100, 750, "MultiModal RAG 系统测试文档")
    c.drawString(100, 700, "第一章：人工智能基础")
    c.drawString(100, 650, "人工智能（AI）是计算机科学的一个重要分支。")
    c.drawString(100, 600, "图1展示了AI的发展历程。")
    c.showPage()
    c.drawString(100, 750, "第二章：机器学习")
    c.drawString(100, 700, "机器学习是AI的核心技术之一。")
    c.drawString(100, 650, "深度学习模型在图像识别领域取得了巨大成功。")
    c.showPage()
    c.drawString(100, 750, "第三章：多模态学习")
    c.drawString(100, 700, "多模态学习涉及文本、图像、语音等多种数据类型。")
    c.showPage()
    c.save()
    return pdf_path
