from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


class TestChatAPI:
    def test_chat_validation(self, client):
        resp = client.post("/api/chat", json={})
        assert resp.status_code == 422

    def test_chat_minimal_request(self, client):
        with patch("app.services.chat.generate_answer") as mock_gen:
            mock_gen.return_value = {
                "answer": "test answer",
                "sources": [],
                "model": "test-model",
                "response_time_ms": 100,
            }
            resp = client.post("/api/chat", json={"query": "test"})
            assert resp.status_code == 200
            data = resp.json()
            assert "answer" in data
            assert "sources" in data

    def test_chat_with_sources(self, client):
        mock_sources = [
            {"text": "相关片段", "file_name": "test.pdf", "chunk_index": 0, "relevance_score": 0.95},
        ]
        with patch("app.services.chat.generate_answer") as mock_gen:
            mock_gen.return_value = {
                "answer": "基于文档的答案",
                "sources": mock_sources,
                "model": "Qwen-VL-Chat",
                "response_time_ms": 3500,
            }
            resp = client.post("/api/chat", json={"query": "这是什么？", "top_k": 5})
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["sources"]) == 1
            assert data["sources"][0]["relevance_score"] == 0.95
            assert data["response_time_ms"] == 3500
