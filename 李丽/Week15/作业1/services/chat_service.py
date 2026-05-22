"""
聊天问答服务
处理用户提问，执行 RAG 流程
"""
import re
import openai
import config
from services.embedding_service import encode_text
from utils.milvus_client import search_vectors

# 初始化 OpenAI 客户端
_client = openai.OpenAI(
    api_key=config.OPENAI_API_KEY,
    base_url=config.OPENAI_BASE_URL
)


def retrieve_context(question: str, top_k: int = None) -> list:
    """
    检索相关上下文

    Args:
        question: 用户问题
        top_k: 检索数量

    Returns:
        检索结果列表
    """
    if top_k is None:
        top_k = config.RAG_TOP_K

    # 1. 编码问题
    query_vector = encode_text(question)

    # 2. 向量检索
    results = search_vectors(query_vector, top_k=top_k)

    return results


def format_context(results: list) -> str:
    """
    格式化检索结果为上下文

    Args:
        results: 检索结果

    Returns:
        格式化的上下文字符串
    """
    context_parts = []

    for i, result in enumerate(results, 1):
        text = result.get("text", "")
        file_name = result.get("file_name", "unknown")

        # 处理图片路径转换
        # 原始: images/xxx.png
        # 转换: ./processed/{file_dir}/vlm/images/xxx.png
        if "images/" in text:
            file_dir = file_name.rsplit(".", 1)[0] if "." in file_name else file_name
            text = text.replace("images/", f"./processed/{file_dir}/vlm/images/")

        context_parts.append(f"[来源 {i}: {file_name}]\n{text}")

    return "\n\n".join(context_parts)


def generate_answer(question: str, context: str) -> str:
    """
    调用大模型生成答案

    Args:
        question: 用户问题
        context: 相关上下文

    Returns:
        生成的答案
    """
    prompt = config.RAG_PROMPT_TEMPLATE.format(
        question=question,
        context=context
    )

    try:
        response = _client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[
                {"role": "system", "content": "你是一个专业的助手，基于提供的资料回答问题。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating answer: {e}")
        return f"抱歉，生成答案时出现错误: {str(e)}"


def chat(question: str) -> dict:
    """
    处理聊天问答请求

    Args:
        question: 用户问题

    Returns:
        包含答案和来源的响应
    """
    # 1. 检索相关上下文
    retrieval_results = retrieve_context(question)

    if not retrieval_results:
        return {
            "code": 0,
            "data": {
                "answer": "抱歉，没有找到相关资料来回答这个问题。",
                "sources": []
            }
        }

    # 2. 格式化上下文
    context = format_context(retrieval_results)

    # 3. 生成答案
    answer = generate_answer(question, context)

    # 4. 格式化来源
    sources = [
        {
            "text": r.get("text", "")[:200] + "...",  # 截断显示
            "file_name": r.get("file_name"),
            "score": round(r.get("score", 0), 4)
        }
        for r in retrieval_results
    ]

    return {
        "code": 0,
        "data": {
            "answer": answer,
            "sources": sources
        }
    }
