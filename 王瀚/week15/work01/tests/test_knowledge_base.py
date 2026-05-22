"""
测试知识库管理

测试逻辑:
1. 创建知识库 → 验证返回字段
2. 列出知识库 → 验证包含刚创建的
3. 获取知识库详情 → 验证数据一致
4. 删除知识库 → 验证删除成功
5. 删除不存在的知识库 → 验证 404
"""

import pytest
from fastapi.testclient import TestClient


class TestKnowledgeBase:
    BASE_URL = "/api/v1/knowledge-bases"

    @classmethod
    def setup_class(cls):
        cls.kb_id = None

    def test_create_kb(self, client: TestClient):
        resp = client.post(self.BASE_URL, json={
            "name": "新知识库",
            "description": "测试描述",
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "新知识库"
        assert data["description"] == "测试描述"
        assert "id" in data
        assert "created_at" in data
        type(self).kb_id = data["id"]

    def test_list_kbs(self, client: TestClient):
        assert type(self).kb_id is not None
        resp = client.get(self.BASE_URL)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 1
        assert any(kb["id"] == type(self).kb_id for kb in data["items"])

    def test_get_kb(self, client: TestClient):
        assert type(self).kb_id is not None
        resp = client.get(f"{self.BASE_URL}/{type(self).kb_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "新知识库"
        assert data["id"] == type(self).kb_id

    def test_get_nonexistent_kb(self, client: TestClient):
        resp = client.get(f"{self.BASE_URL}/nonexistent")
        assert resp.status_code == 404

    def test_delete_kb(self, client: TestClient):
        assert type(self).kb_id is not None
        resp = client.delete(f"{self.BASE_URL}/{type(self).kb_id}")
        assert resp.status_code == 204

    def test_delete_nonexistent_kb(self, client: TestClient):
        resp = client.delete(f"{self.BASE_URL}/nonexistent")
        assert resp.status_code == 404
