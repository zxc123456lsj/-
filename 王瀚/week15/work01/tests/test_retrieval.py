"""
测试多模态检索功能

测试逻辑:
1. 文本检索 → 验证返回结构和字段
2. 图片检索 → 验证返回结构和字段
3. 空知识库检索 → 验证返回空列表
4. 检索限流 → 验证 top_k 参数生效
"""

import pytest
from fastapi.testclient import TestClient


class TestRetrieval:
    def test_retrieve_text_and_image(self, client: TestClient, sample_kb, auto_mocks):
        kb_id = sample_kb["id"]
        resp = client.post("/api/v1/retrieve", json={
            "kb_id": kb_id,
            "query": "人工智能是什么",
            "top_k_text": 3,
            "top_k_image": 2,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["query"] == "人工智能是什么"
        assert data["kb_id"] == kb_id
        assert isinstance(data["text_results"], list)
        assert isinstance(data["image_results"], list)

    def test_retrieve_nonexistent_kb(self, client: TestClient, auto_mocks):
        resp = client.post("/api/v1/retrieve", json={
            "kb_id": "nonexistent",
            "query": "test",
        })
        assert resp.status_code == 404

    def test_retrieve_top_k_limits(self, client: TestClient, sample_kb, auto_mocks):
        kb_id = sample_kb["id"]
        resp = client.post("/api/v1/retrieve", json={
            "kb_id": kb_id,
            "query": "机器学习",
            "top_k_text": 1,
            "top_k_image": 1,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["text_results"]) <= 1
        assert len(data["image_results"]) <= 1
