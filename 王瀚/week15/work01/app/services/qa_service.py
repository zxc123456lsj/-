import json
import os
from typing import Optional
from openai import OpenAI

from app.config import settings
from app.services.retriever import Retriever
from app.core.database import Session
from app.core.models import Document


class QAService:
    def __init__(self):
        self.client: Optional[OpenAI] = None

    def _get_client(self) -> OpenAI:
        if self.client is None:
            self.client = OpenAI(
                api_key=os.getenv("QWEN_API_KEY", ""),
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
        return self.client

    @staticmethod
    def _build_context(text_results: list[dict], image_results: list[dict], kb_id: str) -> tuple[str, list[dict]]:
        db = Session()
        sources = []
        context_parts = []
        seen_docs = {}

        for r in text_results:
            doc = db.query(Document).filter(Document.id == r["doc_id"]).first()
            filename = doc.filename if doc else "unknown"
            source_key = f"{r['doc_id']}_{r['page_num']}"
            if source_key not in seen_docs:
                seen_docs[source_key] = {
                    "doc_id": r["doc_id"],
                    "filename": filename,
                    "page_num": r["page_num"],
                    "chunk_id": r["chunk_id"],
                    "image_id": None,
                }
            context_parts.append(f"[Page {r['page_num']}] {r['content']}")

        for r in image_results:
            doc = db.query(Document).filter(Document.id == r["doc_id"]).first()
            filename = doc.filename if doc else "unknown"
            seen_docs[f"{r['doc_id']}_{r['page_num']}_img"] = {
                "doc_id": r["doc_id"],
                "filename": filename,
                "page_num": r["page_num"],
                "chunk_id": None,
                "image_id": r["image_id"],
            }
            context_parts.append(f"[Page {r['page_num']} Image] {r['file_path']}")

        db.close()
        context = "\n\n".join(context_parts)
        sources = list(seen_docs.values())
        return context, sources

    def answer(self, query: str, kb_id: str) -> tuple[str, list[dict]]:
        text_results, image_results = Retriever.retrieve(query, kb_id)
        context, sources = self._build_context(text_results, image_results, kb_id)
        answer = self._call_qwen(query, context, image_results)
        return answer, sources

    def _call_qwen(self, query: str, context: str, image_results: list[dict]) -> str:
        try:
            client = self._get_client()
            messages = [
                {
                    "role": "system",
                    "content": "你是一个多模态文档问答助手。请基于提供的上下文（文本和图片）回答用户问题。"
                               "回答需要准确、简洁，并明确指出信息来源的页码。如果上下文中没有足够信息，请如实说明。",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"## 上下文\n\n{context}\n\n## 问题\n\n{query}"},
                    ],
                },
            ]
            for img in image_results[:2]:
                file_path = img["file_path"]
                if os.path.isfile(file_path):
                    import base64
                    with open(file_path, "rb") as f:
                        b64 = base64.b64encode(f.read()).decode("utf-8")
                    messages[1]["content"].append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    })

            resp = client.chat.completions.create(
                model=settings.qwen_vl_model,
                messages=messages,
                temperature=0.3,
                max_tokens=1024,
            )
            return resp.choices[0].message.content
        except Exception as e:
            return f"[QA Error: {e}]\n\n基于检索到的上下文，以下是回答：\n{context[:500]}..."
