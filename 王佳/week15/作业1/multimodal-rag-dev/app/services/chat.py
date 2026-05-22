import os
import time

from dotenv import load_dotenv
from openai import OpenAI

from app.services.retriever import retrieve

load_dotenv()

QWEN_VL_MODEL = os.getenv("QWEN_VL_MODEL", "Qwen/Qwen-VL-Chat")

CHAT_PROMPT_TEMPLATE = """你是一个专业的文档分析助手。请基于以下参考文档片段回答用户的问题。

## 要求：
1. 仅基于提供的参考内容回答，不要编造信息
2. 如果参考内容不足以回答问题，请如实说明
3. 回答时请引用具体的来源（文件名和片段编号）
4. 回答要简洁、准确、专业

## 参考文档片段：
{context}

## 用户问题：
{query}

## 回答："""


def generate_answer(query: str, top_k: int = 5) -> dict:
    t0 = time.time()

    retrieved = retrieve(query, top_k=top_k)

    context_parts: list[str] = []
    sources: list[dict] = []
    for i, item in enumerate(retrieved):
        ctx = f"[片段{i + 1}] 文件: {item.get('file_name', 'unknown')}, Chunk: {item.get('chunk_index', 0)}\n{item.get('text', '')}"
        context_parts.append(ctx)
        sources.append({
            "text": item.get("text", "")[:200],
            "file_name": item.get("file_name", ""),
            "chunk_index": item.get("chunk_index", 0),
            "relevance_score": round(item.get("score", 0), 4),
        })

    context = "\n\n".join(context_parts)
    prompt = CHAT_PROMPT_TEMPLATE.format(context=context, query=query)

    try:
        client = OpenAI(
            base_url=os.getenv("QWEN_VL_BASE_URL", "http://localhost:8000/v1"),
            api_key=os.getenv("QWEN_VL_API_KEY", "not-needed"),
        )
        response = client.chat.completions.create(
            model=QWEN_VL_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.3,
        )
        answer = response.choices[0].message.content or ""
    except Exception:
        answer = f"基于 {len(retrieved)} 个参考片段，无法连接到推理服务。请检查 Qwen-VL 服务是否正常。"

    elapsed_ms = int((time.time() - t0) * 1000)

    return {
        "answer": answer,
        "sources": sources,
        "model": QWEN_VL_MODEL,
        "response_time_ms": elapsed_ms,
    }
