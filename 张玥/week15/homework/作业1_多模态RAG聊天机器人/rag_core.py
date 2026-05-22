"""多模态 RAG 聊天机器人作业版核心逻辑。

这份代码保留课程项目中的接口语义，但把 Kafka、Milvus、云端模型替换为本地轻量实现。
它适合用于作业提交、流程讲解和后续继续重构。
"""

from __future__ import annotations

import os
import re
import shutil
import sqlite3
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = Path(__file__).resolve().parent
STORAGE_DIR = BASE_DIR / "storage"
UPLOAD_DIR = STORAGE_DIR / "uploads"
PROCESSED_DIR = STORAGE_DIR / "processed"
DB_PATH = STORAGE_DIR / "rag_homework.db"


@dataclass
class DocumentRecord:
    """SQLite 中保存的文档元信息。"""

    id: int
    filename: str
    filepath: str
    status: str
    parsed_path: str | None = None


@dataclass
class Chunk:
    """可检索的文档片段。"""

    document_id: int
    filename: str
    text: str
    images: list[str]


@dataclass
class SearchHit:
    """检索命中的片段和分数。"""

    chunk: Chunk
    score: float


def init_storage() -> None:
    """初始化本地目录和 SQLite 表。"""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                filepath TEXT NOT NULL,
                status TEXT NOT NULL,
                parsed_path TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()


def _row_to_record(row: tuple) -> DocumentRecord:
    return DocumentRecord(
        id=row[0],
        filename=row[1],
        filepath=row[2],
        status=row[3],
        parsed_path=row[4],
    )


def get_document(document_id: int) -> DocumentRecord:
    """根据 ID 查询文档记录。"""
    init_storage()
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT id, filename, filepath, status, parsed_path FROM documents WHERE id = ?",
            (document_id,),
        ).fetchone()

    if row is None:
        raise ValueError(f"文档不存在：{document_id}")

    return _row_to_record(row)


def upload_document(source_path: Path) -> DocumentRecord:
    """上传本地文件，保存副本并记录状态。

    这对应原项目中的 `post /upload/document`。
    """
    init_storage()
    source_path = Path(source_path)
    if not source_path.exists():
        raise FileNotFoundError(f"上传文件不存在：{source_path}")

    suffix = source_path.suffix or ".txt"
    save_name = f"{uuid.uuid4().hex}{suffix}"
    target_path = UPLOAD_DIR / save_name
    shutil.copy2(source_path, target_path)

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(
            "INSERT INTO documents(filename, filepath, status) VALUES (?, ?, ?)",
            (source_path.name, str(target_path), "uploaded"),
        )
        conn.commit()
        document_id = int(cursor.lastrowid)

    return get_document(document_id)


def upload_document_bytes(filename: str, content: bytes) -> DocumentRecord:
    """上传接口使用的 bytes 版本，便于 FastAPI 接收文件。"""
    init_storage()
    suffix = Path(filename).suffix or ".txt"
    save_name = f"{uuid.uuid4().hex}{suffix}"
    target_path = UPLOAD_DIR / save_name
    target_path.write_bytes(content)

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(
            "INSERT INTO documents(filename, filepath, status) VALUES (?, ?, ?)",
            (filename, str(target_path), "uploaded"),
        )
        conn.commit()
        document_id = int(cursor.lastrowid)

    return get_document(document_id)


def parse_document_task(document_id: int, use_mineru: bool = False) -> DocumentRecord:
    """解析文档并生成 Markdown。

    这对应原项目中的离线 worker。作业版默认不用 MinerU，避免环境过重；
    如果本地已安装 mineru 并传入 use_mineru=True，会尝试调用 mineru。
    """
    record = get_document(document_id)
    source_path = Path(record.filepath)
    output_dir = PROCESSED_DIR / str(document_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    parsed_path = output_dir / "content.md"

    if use_mineru and shutil.which("mineru"):
        subprocess.run(
            ["mineru", "-p", str(source_path), "-o", str(output_dir)],
            check=True,
            timeout=600,
        )
        mineru_outputs = list(output_dir.rglob("*.md"))
        if mineru_outputs:
            parsed_path = mineru_outputs[0]
    elif source_path.suffix.lower() in {".md", ".txt"}:
        text = source_path.read_text(encoding="utf-8", errors="ignore")
        parsed_path.write_text(text, encoding="utf-8")
        _copy_optional_assets(source_path.parent / "images", output_dir / "images")
    elif source_path.suffix.lower() == ".pdf":
        text = _extract_pdf_text(source_path)
        parsed_path.write_text(f"# {record.filename}\n\n{text}", encoding="utf-8")
    else:
        text = source_path.read_text(encoding="utf-8", errors="ignore")
        parsed_path.write_text(f"# {record.filename}\n\n{text}", encoding="utf-8")

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "UPDATE documents SET status = ?, parsed_path = ? WHERE id = ?",
            ("parsed", str(parsed_path), document_id),
        )
        conn.commit()

    return get_document(document_id)


def _extract_pdf_text(pdf_path: Path) -> str:
    """PDF 解析兜底方案：优先使用 pdfplumber，失败时返回提示。"""
    try:
        import pdfplumber

        pages = []
        with pdfplumber.open(pdf_path) as pdf:
            for index, page in enumerate(pdf.pages[:10], start=1):
                text = page.extract_text() or ""
                pages.append(f"## 第 {index} 页\n\n{text}")
        return "\n\n".join(pages)
    except Exception as exc:
        return f"PDF 文本抽取失败：{exc}"


def _copy_optional_assets(source_dir: Path, target_dir: Path) -> None:
    """复制 Markdown 旁边的图片目录，保证图文链接更接近真实项目。"""
    if source_dir.exists() and source_dir.is_dir():
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.copytree(source_dir, target_dir)


def load_parsed_chunks() -> list[Chunk]:
    """加载所有已解析文档，并切分成可检索片段。"""
    init_storage()
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            """
            SELECT id, filename, filepath, status, parsed_path
            FROM documents
            WHERE status IN ('parsed', 'indexed') AND parsed_path IS NOT NULL
            """
        ).fetchall()

    chunks: list[Chunk] = []
    for row in rows:
        record = _row_to_record(row)
        parsed_path = Path(record.parsed_path or "")
        if not parsed_path.exists():
            continue
        text = parsed_path.read_text(encoding="utf-8", errors="ignore")
        for part in split_markdown(text):
            images = extract_markdown_images(part)
            chunks.append(
                Chunk(
                    document_id=record.id,
                    filename=record.filename,
                    text=part,
                    images=images,
                )
            )
    return chunks


def split_markdown(markdown_text: str, max_chars: int = 500) -> list[str]:
    """按段落切分 Markdown，过长段落再按字符长度切分。"""
    paragraphs = [item.strip() for item in markdown_text.split("\n\n") if item.strip()]
    result: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= max_chars:
            result.append(paragraph)
            continue
        for start in range(0, len(paragraph), max_chars):
            result.append(paragraph[start : start + max_chars].strip())
    return result


def extract_markdown_images(markdown_text: str) -> list[str]:
    """抽取 Markdown 图片链接。"""
    return re.findall(r"!\[.*?\]\((.*?)\)", markdown_text)


def build_index() -> dict:
    """构建本地检索索引。

    作业版每次调用时重新从已解析文档构建 TF-IDF 索引，避免引入向量数据库。
    """
    chunks = load_parsed_chunks()
    if not chunks:
        return {"chunks": [], "vectorizer": None, "matrix": None}

    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4))
    matrix = vectorizer.fit_transform([chunk.text for chunk in chunks])

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("UPDATE documents SET status = 'indexed' WHERE status = 'parsed'")
        conn.commit()

    return {"chunks": chunks, "vectorizer": vectorizer, "matrix": matrix}


def retrieve(question: str, top_k: int = 3) -> list[SearchHit]:
    """根据问题检索相关片段。"""
    index = build_index()
    chunks: list[Chunk] = index["chunks"]
    vectorizer = index["vectorizer"]
    matrix = index["matrix"]
    if not chunks or vectorizer is None or matrix is None:
        return []

    query_vector = vectorizer.transform([question])
    scores = cosine_similarity(query_vector, matrix).flatten()
    ranked_indexes = scores.argsort()[::-1][:top_k]

    return [SearchHit(chunk=chunks[i], score=float(scores[i])) for i in ranked_indexes]


def chat_with_rag(question: str, top_k: int = 3) -> dict:
    """检索资料并生成作业版回答。

    如果后续要接真实 LLM，可以把这里的模板回答替换为模型调用。
    """
    hits = retrieve(question, top_k=top_k)
    if not hits:
        return {
            "answer": "当前没有可检索资料，请先上传并解析文档。",
            "hits": [],
        }

    context_lines = []
    image_lines = []
    for index, hit in enumerate(hits, start=1):
        context_lines.append(
            f"资料 {index}（文件：{hit.chunk.filename}，相关度：{hit.score:.4f}）：\n{hit.chunk.text}"
        )
        for image in hit.chunk.images:
            image_lines.append(f"![来源图片]({image})")

    answer = (
        f"根据已检索到的资料，问题“{question}”可以这样回答：\n\n"
        + "\n\n".join(context_lines)
    )
    if image_lines:
        answer += "\n\n相关图片：\n" + "\n".join(dict.fromkeys(image_lines))

    return {
        "answer": answer,
        "hits": [
            {
                "filename": hit.chunk.filename,
                "score": hit.score,
                "text": hit.chunk.text,
                "images": hit.chunk.images,
            }
            for hit in hits
        ],
    }


def render_markdown_with_images(markdown_text: str) -> list[dict]:
    """把 Markdown 文本拆成文本片段和图片片段，便于前端分别渲染。"""
    pattern = re.compile(r"!\[(.*?)\]\((.*?)\)")
    parts: list[dict] = []
    last_pos = 0

    for match in pattern.finditer(markdown_text):
        if match.start() > last_pos:
            text = markdown_text[last_pos : match.start()].strip()
            if text:
                parts.append({"type": "text", "content": text})
        parts.append({"type": "image", "alt": match.group(1), "url": match.group(2)})
        last_pos = match.end()

    tail = markdown_text[last_pos:].strip()
    if tail:
        parts.append({"type": "text", "content": tail})

    return parts


def reset_demo_storage() -> None:
    """仅用于本地 Demo：清空作业版 storage。"""
    if STORAGE_DIR.exists():
        shutil.rmtree(STORAGE_DIR)
    init_storage()
