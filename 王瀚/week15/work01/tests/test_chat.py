"""
测试多模态问答功能

测试逻辑:
1. 问答请求 → 验证返回结构和字段完整性
2. 不存在的知识库 → 验证 404
3. 空问题 → 验证参数校验返回 422
4. QA 服务单元测试 → 验证 sources 结构
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app.services.qa_service import QAService
from app.services.embedding_service import EmbeddingService


class TestChat:
    def test_chat_with_answer(self, client: TestClient, sample_kb, auto_mocks):
        kb_id = sample_kb["id"]
        with patch.object(QAService, "_call_qwen", return_value="人工智能是计算机科学的分支。"):
            resp = client.post("/api/v1/chat", json={
                "kb_id": kb_id,
                "query": "什么是人工智能？",
            })
        assert resp.status_code == 200
        data = resp.json()
        assert data["query"] == "什么是人工智能？"
        assert "人工智能" in data["answer"]
        assert isinstance(data["sources"], list)
        assert isinstance(data["answer"], str)

    def test_chat_nonexistent_kb(self, client: TestClient, auto_mocks):
        resp = client.post("/api/v1/chat", json={
            "kb_id": "nonexistent",
            "query": "test",
        })
        assert resp.status_code == 404

    def test_chat_empty_query(self, client: TestClient, sample_kb, auto_mocks):
        kb_id = sample_kb["id"]
        resp = client.post("/api/v1/chat", json={
            "kb_id": kb_id,
            "query": "",
        })
        assert resp.status_code == 422


class TestQAServiceUnit:
    def test_answer_calls_qwen(self):
        qa = QAService()
        with patch.object(EmbeddingService, "embed_text_bge", return_value=[0.0] * 1024), \
             patch.object(EmbeddingService, "embed_text_clip", return_value=[0.0] * 512), \
             patch.object(qa, "_call_qwen", return_value="Mock answer"):
            answer, sources = qa.answer("测试问题", "test-kb")
        assert answer == "Mock answer"
        assert isinstance(sources, list)

    def test_build_context_structure(self):
        qa = QAService()
        text_results = [
            {"doc_id": "d1", "chunk_id": "c1", "content": "AI text", "page_num": 1, "score": 0.9},
        ]
        image_results = [
            {"image_id": "i1", "doc_id": "d1", "file_path": "/path/img.png", "page_num": 2, "score": 0.8},
        ]
        context, sources = qa._build_context(text_results, image_results, "kb1")
        assert "AI text" in context
        assert len(sources) > 0
        for s in sources:
            assert "doc_id" in s
            assert "page_num" in s
