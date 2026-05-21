"""MinerU 与 pdfplumber 文档解析对比脚本。

脚本会使用 pdfplumber 抽取目标 PDF 的文本摘要，并检测本机是否安装 mineru。
如果 mineru 可用，则尝试调用 MinerU 解析；如果不可用，只记录安装建议。
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import pdfplumber


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PDF = BASE_DIR.parent.parent / "模型论文" / "2509-MinerU2.5.pdf"
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs"


def parse_with_pdfplumber(pdf_path: Path, output_dir: Path, max_pages: int = 3) -> dict:
    """使用 pdfplumber 抽取前几页文本，并保存摘要。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "pdfplumber_result.md"

    page_texts: list[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        page_count = len(pdf.pages)
        for index, page in enumerate(pdf.pages[:max_pages], start=1):
            text = page.extract_text() or ""
            page_texts.append(f"## 第 {index} 页\n\n{text[:2000]}")

    content = "\n\n".join(page_texts)
    result_path.write_text(
        f"# pdfplumber 解析结果摘要\n\n"
        f"- PDF 文件：`{pdf_path}`\n"
        f"- 总页数：{page_count}\n"
        f"- 本次抽取页数：{min(max_pages, page_count)}\n"
        f"- 抽取文本长度：{len(content)}\n\n"
        f"{content}\n",
        encoding="utf-8",
    )
    return {
        "tool": "pdfplumber",
        "available": True,
        "page_count": page_count,
        "sample_pages": min(max_pages, page_count),
        "text_length": len(content),
        "result_path": str(result_path),
    }


def parse_with_mineru(pdf_path: Path, output_dir: Path) -> dict:
    """检测并尝试使用 MinerU 解析 PDF。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "mineru_status.md"
    mineru_cli = shutil.which("mineru")

    if not mineru_cli:
        status_path.write_text(
            "# MinerU 解析状态\n\n"
            "- 当前环境未检测到 `mineru` CLI。\n"
            "- 为避免污染现有 `py312` 环境，本脚本不自动安装 MinerU。\n"
            "- 可按课程说明手动安装：\n\n"
            "```powershell\n"
            "pip install uv -i https://mirrors.aliyun.com/pypi/simple\n"
            "uv pip install -U \"mineru[core]\" -i https://mirrors.aliyun.com/pypi/simple\n"
            "```\n",
            encoding="utf-8",
        )
        return {
            "tool": "mineru",
            "available": False,
            "status_path": str(status_path),
        }

    mineru_output = output_dir / "mineru_output"
    command = ["mineru", "-p", str(pdf_path), "-o", str(mineru_output)]
    try:
        subprocess.run(command, check=True, timeout=900)
        markdown_files = list(mineru_output.rglob("*.md"))
        status_path.write_text(
            "# MinerU 解析状态\n\n"
            f"- 命令：`{' '.join(command)}`\n"
            f"- 解析成功：是\n"
            f"- Markdown 文件数量：{len(markdown_files)}\n"
            f"- 输出目录：`{mineru_output}`\n",
            encoding="utf-8",
        )
        return {
            "tool": "mineru",
            "available": True,
            "success": True,
            "markdown_count": len(markdown_files),
            "status_path": str(status_path),
        }
    except Exception as exc:
        status_path.write_text(
            "# MinerU 解析状态\n\n"
            f"- 命令：`{' '.join(command)}`\n"
            f"- 解析成功：否\n"
            f"- 错误：`{exc}`\n",
            encoding="utf-8",
        )
        return {
            "tool": "mineru",
            "available": True,
            "success": False,
            "error": str(exc),
            "status_path": str(status_path),
        }


def write_summary(pdfplumber_result: dict, mineru_result: dict, output_dir: Path) -> Path:
    """写出本次运行记录。"""
    summary_path = output_dir / "解析对比运行记录.md"
    summary_path.write_text(
        "# 解析对比运行记录\n\n"
        "## pdfplumber\n\n"
        f"- 可用：{pdfplumber_result['available']}\n"
        f"- PDF 总页数：{pdfplumber_result['page_count']}\n"
        f"- 抽取页数：{pdfplumber_result['sample_pages']}\n"
        f"- 文本长度：{pdfplumber_result['text_length']}\n"
        f"- 结果文件：`{pdfplumber_result['result_path']}`\n\n"
        "## MinerU\n\n"
        f"- 可用：{mineru_result.get('available')}\n"
        f"- 状态文件：`{mineru_result.get('status_path')}`\n\n"
        "## 初步结论\n\n"
        "- pdfplumber 已成功抽取文本，适合简单文本层 PDF 的轻量读取。\n"
        "- MinerU 当前是否可实际运行取决于本机 CLI 和模型依赖；它更适合复杂 PDF 的结构化 Markdown 解析。\n",
        encoding="utf-8",
    )
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对比 pdfplumber 与 MinerU 的 PDF 解析效果")
    parser.add_argument("--pdf", default=str(DEFAULT_PDF), help="待解析 PDF 路径")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="输出目录")
    parser.add_argument("--max-pages", type=int, default=3, help="pdfplumber 抽取前几页作为摘要")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pdf_path = Path(args.pdf)
    output_dir = Path(args.output_dir)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF 文件不存在：{pdf_path}")

    pdfplumber_result = parse_with_pdfplumber(pdf_path, output_dir, max_pages=args.max_pages)
    mineru_result = parse_with_mineru(pdf_path, output_dir)
    summary_path = write_summary(pdfplumber_result, mineru_result, output_dir)

    print("pdfplumber 解析完成：", pdfplumber_result)
    print("MinerU 状态：", mineru_result)
    print("运行记录：", summary_path)


if __name__ == "__main__":
    main()
