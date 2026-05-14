"""第14周作业 1：LangChain 本地知识库问答。

本脚本演示一个最小可用的 RAG 流程：
1. 读取本地知识库文档
2. 将文档切分成较小片段
3. 用 TF-IDF 做轻量检索
4. 把检索结果作为上下文交给大模型回答

如果没有配置 API Key，程序会跳过大模型调用，只展示检索结果。
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_DIR = BASE_DIR / "knowledge_base"
DEFAULT_QUESTIONS = [
    "LangChain 主要解决什么问题？",
    "LangGraph 和普通链式调用有什么区别？",
    "Deep Agents 适合什么样的任务？",
]


@dataclass
class DocumentChunk:
    """本地知识库中的一个文档片段。"""

    source: str
    content: str


def load_documents(knowledge_dir: Path) -> list[DocumentChunk]:
    """读取本地知识库中的 Markdown 和文本文件。"""
    if not knowledge_dir.exists():
        raise FileNotFoundError(f"知识库目录不存在：{knowledge_dir}")

    chunks: list[DocumentChunk] = []
    for file_path in sorted(knowledge_dir.glob("*")):
        if file_path.suffix.lower() not in {".md", ".txt"}:
            continue

        text = file_path.read_text(encoding="utf-8")
        chunks.extend(split_text(file_path.name, text))

    if not chunks:
        raise ValueError(f"知识库目录中没有可检索的 .md 或 .txt 文档：{knowledge_dir}")

    return chunks


def split_text(source: str, text: str, max_chars: int = 320) -> list[DocumentChunk]:
    """按段落切分文档，避免一次塞给模型的上下文过长。"""
    paragraphs = [item.strip() for item in text.split("\n\n") if item.strip()]
    chunks: list[DocumentChunk] = []

    for paragraph in paragraphs:
        if len(paragraph) <= max_chars:
            chunks.append(DocumentChunk(source=source, content=paragraph))
            continue

        # 段落过长时按固定长度继续切分，保证每个 chunk 都比较短。
        for start in range(0, len(paragraph), max_chars):
            part = paragraph[start : start + max_chars].strip()
            if part:
                chunks.append(DocumentChunk(source=source, content=part))

    return chunks


def retrieve(question: str, chunks: list[DocumentChunk], top_k: int = 3) -> list[tuple[DocumentChunk, float]]:
    """使用字符级 TF-IDF 检索最相关的文档片段。

    中文文本没有天然空格分词，所以这里使用字符 n-gram，初学阶段更稳定。
    """
    corpus = [chunk.content for chunk in chunks]
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4))
    matrix = vectorizer.fit_transform(corpus)
    question_vector = vectorizer.transform([question])
    scores = cosine_similarity(question_vector, matrix).flatten()

    ranked_indexes = scores.argsort()[::-1][:top_k]
    return [(chunks[index], float(scores[index])) for index in ranked_indexes]


def build_prompt(question: str, retrieved_chunks: Iterable[tuple[DocumentChunk, float]]) -> str:
    """把检索结果组装成大模型可理解的提示词。"""
    context_lines = []
    for index, (chunk, score) in enumerate(retrieved_chunks, start=1):
        context_lines.append(
            f"【片段 {index}｜来源：{chunk.source}｜相关度：{score:.4f}】\n{chunk.content}"
        )

    context = "\n\n".join(context_lines)
    return f"""你是一个面向初学者的 AI 学习助手。
请只根据下面的本地知识库内容回答问题。如果知识库没有提供依据，请说明“当前知识库信息不足”。

本地知识库内容：
{context}

用户问题：
{question}

请用中文回答，语言要清晰、适合 Java 转 Python/AI 的学习者理解。
"""


def answer_with_llm(prompt: str) -> str | None:
    """调用 LangChain ChatOpenAI 生成回答，未配置 Key 时返回 None。"""
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        return None

    base_url = os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    model_name = os.getenv("OPENAI_MODEL", "qwen-flash")

    from langchain_openai import ChatOpenAI

    model = ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=0.2,
    )
    response = model.invoke(prompt)
    return str(response.content)


def run_question(question: str, top_k: int) -> None:
    """执行一次完整问答流程。"""
    chunks = load_documents(KNOWLEDGE_DIR)
    retrieved_chunks = retrieve(question, chunks, top_k=top_k)
    prompt = build_prompt(question, retrieved_chunks)
    answer = answer_with_llm(prompt)

    print("=" * 80)
    print(f"用户问题：{question}")
    print("\n检索命中的文档片段：")
    for index, (chunk, score) in enumerate(retrieved_chunks, start=1):
        print(f"\n{index}. 来源：{chunk.source}｜相关度：{score:.4f}")
        print(chunk.content)

    print("\n最终回答：")
    if answer is None:
        print("未检测到 OPENAI_API_KEY 或 DASHSCOPE_API_KEY，已跳过 LLM 调用。")
        print("你可以先检查上面的检索结果；配置 API Key 后会生成自然语言回答。")
    else:
        print(answer)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LangChain 本地知识库问答作业")
    parser.add_argument("--question", help="自定义问题；不传则运行内置演示问题")
    parser.add_argument("--top-k", type=int, default=3, help="检索返回的片段数量")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    questions = [args.question] if args.question else DEFAULT_QUESTIONS
    for question in questions:
        run_question(question, top_k=args.top_k)


if __name__ == "__main__":
    main()
