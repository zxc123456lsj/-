from unittest.mock import patch

import pytest


class TestRetriever:
    def test_retrieve_returns_results(self):
        mock_results = [
            {"text": "test chunk", "file_name": "test.pdf", "chunk_index": 0, "score": 0.95},
        ]
        with patch("app.services.retriever.search_text", return_value=mock_results):
            with patch("app.services.retriever.search_image", return_value=[]):
                from app.services.retriever import retrieve
                results = retrieve("测试查询", top_k=3)
                assert len(results) <= 3
                assert results[0]["text"] == "test chunk"

    def test_retrieve_merges_text_and_image(self):
        text_results = [
            {"text": "text chunk", "file_name": "a.pdf", "chunk_index": 0, "score": 0.9},
        ]
        image_results = [
            {"text": "[IMAGE]", "file_name": "a.pdf", "chunk_index": 1, "score": 0.85},
        ]
        with patch("app.services.retriever.search_text", return_value=text_results):
            with patch("app.services.retriever.search_image", return_value=image_results):
                from app.services.retriever import retrieve
                results = retrieve("test", top_k=5)
                assert len(results) >= 1

    def test_retrieve_empty_query(self):
        with patch("app.services.retriever.search_text", return_value=[]):
            with patch("app.services.retriever.search_image", return_value=[]):
                from app.services.retriever import retrieve
                results = retrieve("无关联查询", top_k=5)
                assert results == []

    def test_retrieve_respects_top_k(self):
        mock_results = [{"text": f"chunk {i}", "file_name": "f.pdf", "chunk_index": i, "score": 0.9 - i * 0.1} for i in range(10)]
        with patch("app.services.retriever.search_text", return_value=mock_results):
            with patch("app.services.retriever.search_image", return_value=[]):
                from app.services.retriever import retrieve
                results = retrieve("test", top_k=3)
                assert len(results) == 3
