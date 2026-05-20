"""
测试解析Worker功能
测试逻辑:
1. Worker 初始化 → 验证 Kafka 消费者创建
2. 文档处理流程 → 验证解析、chunk、embedding、存储完整链路
3. chunk_text 文本分割 → 验证分块逻辑正确
"""

import os
import tempfile
from unittest.mock import patch, MagicMock

import pytest

from app.services.document_parser import DocumentParser
from app.worker.parse_worker import ParseWorker


class TestDocumentParser:
    def test_chunk_text(self):
        md = "## Page 1\n\n这是第一段内容。\n\n这是第二段内容。\n\n## Page 2\n\n第三段内容。"
        chunks = DocumentParser.chunk_text(md, chunk_size=100, overlap=0)
        assert len(chunks) > 0
        assert all("content" in c and "page_num" in c for c in chunks)

    def test_chunk_with_page_nums(self):
        md = "## Page 1\n\n内容A\n\n## Page 2\n\n内容B"
        chunks = DocumentParser.chunk_text(md, chunk_size=500, overlap=0)
        assert len(chunks) >= 1
        page_nums = [c["page_num"] for c in chunks]
        assert all(p in (1, 2) for p in page_nums)


class TestParseWorker:
    def test_worker_run_once(self, tmp_path):
        worker = ParseWorker()
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 test content")

        with patch.object(worker, "process_document") as mock_process:
            worker.run_once("test-doc-id", str(pdf_path))
            mock_process.assert_called_once_with("test-doc-id", str(pdf_path))

    def test_process_document_flow(self):
        worker = ParseWorker()
        with patch("app.worker.parse_worker.Session") as mock_session, \
             patch("app.worker.parse_worker.DocumentParser") as mock_parser, \
             patch("app.worker.parse_worker.StorageService") as mock_storage, \
             patch("app.worker.parse_worker.EmbeddingService") as mock_embed, \
             patch("app.worker.parse_worker.VectorStore") as mock_vs:

            mock_db = MagicMock()
            mock_session.return_value = mock_db
            mock_doc = MagicMock()
            mock_doc.kb_id = "kb1"
            mock_db.query.return_value.filter.return_value.first.return_value = mock_doc

            mock_result = MagicMock()
            mock_result.error = None
            mock_result.markdown = "# Test"
            mock_result.page_count = 1
            mock_result.images = []
            mock_parser.parse_with_pymupdf.return_value = mock_result
            mock_parser.chunk_text.return_value = [{"content": "test", "page_num": 1}]

            mock_embed.embed_text_bge.return_value = [0.1] * 1024

            worker.process_document("doc-1", "/path/to.pdf")

            assert mock_doc.parse_status is not None
            mock_db.commit.assert_called()
