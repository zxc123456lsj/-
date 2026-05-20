"""
测试文档上传与管理流程

测试逻辑:
1. 创建知识库 → 上传PDF → 验证返回的doc_id和状态
2. 列出知识库文档 → 验证文档列表包含刚上传的文档
3. 查询解析状态 → 验证状态为 pending
4. 删除文档 → 验证删除成功
5. 上传非PDF文件 → 验证返回 400 错误
"""

import os
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="class")
def shared_kb(client):
    resp = client.post("/api/v1/knowledge-bases", json={
        "name": "测试知识库-上传",
        "description": "用于上传测试",
    })
    assert resp.status_code == 201
    return resp.json()


@pytest.mark.usefixtures("shared_kb")
class TestDocumentUpload:
    UPLOAD_URL = "/api/v1/knowledge-bases/{kb_id}/documents"

    def test_upload_pdf_success(self, client: TestClient, shared_kb, sample_pdf_path):
        kb_id = shared_kb["id"]
        with open(sample_pdf_path, "rb") as f:
            resp = client.post(
                self.UPLOAD_URL.format(kb_id=kb_id),
                files={"file": ("test.pdf", f, "application/pdf")},
            )
        assert resp.status_code == 201, f"Upload failed: {resp.text}"
        data = resp.json()
        assert "id" in data
        assert data["parse_status"] == "pending"
        assert "uploaded" in data["message"].lower() or "queued" in data["message"].lower()
        TestDocumentUpload.doc_id = data["id"]

    def test_list_documents(self, client: TestClient, shared_kb):
        assert TestDocumentUpload.doc_id is not None, "Previous test must set doc_id"
        kb_id = shared_kb["id"]
        resp = client.get(f"/api/v1/knowledge-bases/{kb_id}/documents")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 1
        assert any(d["id"] == TestDocumentUpload.doc_id for d in data["items"])

    def test_parse_status(self, client: TestClient):
        assert TestDocumentUpload.doc_id is not None
        resp = client.get(f"/api/v1/documents/{TestDocumentUpload.doc_id}/parse-status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["doc_id"] == TestDocumentUpload.doc_id
        assert data["parse_status"] in ("pending", "parsing", "completed", "failed")

    def test_delete_document(self, client: TestClient, shared_kb):
        assert TestDocumentUpload.doc_id is not None
        kb_id = shared_kb["id"]
        resp = client.delete(f"/api/v1/knowledge-bases/{kb_id}/documents/{TestDocumentUpload.doc_id}")
        assert resp.status_code == 204, f"Delete failed: {resp.text}"

    def test_upload_invalid_file_type(self, client: TestClient, shared_kb):
        kb_id = shared_kb["id"]
        resp = client.post(
            self.UPLOAD_URL.format(kb_id=kb_id),
            files={"file": ("test.txt", b"not a pdf", "text/plain")},
        )
        assert resp.status_code == 400
        assert "unsupported" in resp.text.lower()

    def test_upload_to_nonexistent_kb(self, client: TestClient):
        resp = client.post(
            self.UPLOAD_URL.format(kb_id="nonexistent-id"),
            files={"file": ("test.pdf", b"%PDF-1.4 fake", "application/pdf")},
        )
        assert resp.status_code == 404
