import os
import tempfile

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models.database import Base, engine


@pytest.fixture(autouse=True)
def setup_db():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def sample_pdf():
    """创建一个最小的伪PDF文件用于测试"""
    content = b"%PDF-1.4\n%fake pdf content for testing\n%%EOF"
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp.write(content)
    tmp.close()
    yield tmp.name
    os.unlink(tmp.name)


class TestFileUpload:
    def test_upload_rejects_non_pdf(self, client):
        resp = client.post("/api/upload", files={"file": ("test.txt", b"hello", "text/plain")})
        assert resp.status_code == 400
        assert "仅支持PDF" in resp.json()["detail"]

    def test_upload_accepts_pdf(self, client, sample_pdf):
        with open(sample_pdf, "rb") as f:
            resp = client.post("/api/upload", files={"file": ("test.pdf", f.read(), "application/pdf")})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert "file_id" in data
        assert data["filename"] == "test.pdf"

    def test_upload_dedup(self, client, sample_pdf):
        with open(sample_pdf, "rb") as f:
            content = f.read()
        resp1 = client.post("/api/upload", files={"file": ("test.pdf", content, "application/pdf")})
        resp2 = client.post("/api/upload", files={"file": ("test.pdf", content, "application/pdf")})
        assert resp2.json()["file_id"] == resp1.json()["file_id"]
        assert "已存在" in resp2.json()["message"]

    def test_get_file_status(self, client, sample_pdf):
        with open(sample_pdf, "rb") as f:
            upload_resp = client.post("/api/upload", files={"file": ("test.pdf", f.read(), "application/pdf")})
        file_id = upload_resp.json()["file_id"]
        resp = client.get(f"/api/files/{file_id}/status")
        assert resp.status_code == 200
        assert resp.json()["file_id"] == file_id

    def test_get_file_status_404(self, client):
        resp = client.get("/api/files/99999/status")
        assert resp.status_code == 404

    def test_list_files(self, client, sample_pdf):
        with open(sample_pdf, "rb") as f:
            client.post("/api/upload", files={"file": ("test.pdf", f.read(), "application/pdf")})
        resp = client.get("/api/files")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 1
        assert len(data["files"]) >= 1


class TestHealthCheck:
    def test_health_endpoint(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        assert "status" in resp.json()
        assert "services" in resp.json()
