"""运行多模态 RAG 作业版最小本地流程。"""

from __future__ import annotations

from pathlib import Path

from rag_core import (
    BASE_DIR,
    build_index,
    chat_with_rag,
    parse_document_task,
    render_markdown_with_images,
    reset_demo_storage,
    upload_document,
)


def main() -> None:
    reset_demo_storage()

    sample_path = BASE_DIR / "sample_data" / "sample_report.md"
    record = upload_document(sample_path)
    print(f"上传完成：document_id={record.id}, filename={record.filename}, status={record.status}")

    parsed_record = parse_document_task(record.id)
    print(f"解析完成：document_id={parsed_record.id}, status={parsed_record.status}")
    print(f"解析产物：{parsed_record.parsed_path}")

    index = build_index()
    print(f"索引构建完成：chunk_count={len(index['chunks'])}")

    question = "系统架构图说明了哪些模块？"
    result = chat_with_rag(question)
    print("\n用户问题：", question)
    print("\n回答：")
    print(result["answer"])

    print("\n前端渲染片段：")
    for part in render_markdown_with_images(result["answer"]):
        print(part)


if __name__ == "__main__":
    main()
