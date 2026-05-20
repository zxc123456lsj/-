import os
import json
import tempfile
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class ParseResult:
    markdown: str = ""
    images: list[dict] = field(default_factory=list)
    page_count: int = 0
    error: Optional[str] = None


class DocumentParser:
    @staticmethod
    def parse_with_mineru(pdf_path: str, output_dir: str) -> ParseResult:
        result = ParseResult()
        try:
            from magic_pdf.pipeline import Pipeline
            pipe = Pipeline(pdf_path, output_dir)
            pipe.run()
            md_path = os.path.join(output_dir, "output.md")
            if os.path.exists(md_path):
                with open(md_path, "r", encoding="utf-8") as f:
                    result.markdown = f.read()
            image_dir = os.path.join(output_dir, "images")
            if os.path.isdir(image_dir):
                for idx, img_file in enumerate(sorted(os.listdir(image_dir))):
                    img_path = os.path.join(image_dir, img_file)
                    result.images.append({
                        "file_path": img_path,
                        "page_num": 0,
                        "image_index": idx,
                    })
            result.page_count = len(result.images) if not result.markdown else max(1, result.markdown.count("\n---\n") + 1)
        except Exception as e:
            result.error = str(e)
        return result

    @staticmethod
    def parse_with_pymupdf(pdf_path: str) -> ParseResult:
        import fitz
        result = ParseResult()
        doc = fitz.open(pdf_path)
        result.page_count = len(doc)
        md_parts = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            md_parts.append(f"## Page {page_num + 1}\n\n{page.get_text()}")
            images_on_page = page.get_images(full=True)
            for img_idx, img_info in enumerate(images_on_page):
                xref = img_info[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                ext = base_image["ext"]
                img_name = f"p{page_num + 1}_i{img_idx}.{ext}"
                img_dir = os.path.join(os.path.dirname(pdf_path), "..", "..", "storage", "images")
                os.makedirs(img_dir, exist_ok=True)
                img_path = os.path.join(img_dir, img_name)
                with open(img_path, "wb") as f:
                    f.write(image_bytes)
                result.images.append({
                    "file_path": img_path,
                    "page_num": page_num + 1,
                    "image_index": img_idx,
                })
                md_parts.append(f"\n![Page {page_num + 1} Image {img_idx + 1}]({img_path})\n")
        result.markdown = "\n".join(md_parts)
        doc.close()
        return result

    @staticmethod
    def chunk_text(markdown: str, chunk_size: int = 512, overlap: int = 64) -> list[dict]:
        paragraphs = markdown.split("\n\n")
        chunks = []
        current_chunk = ""
        page_num = 1
        for para in paragraphs:
            if para.startswith("## Page "):
                try:
                    page_num = int(para.replace("## Page ", "").strip())
                except ValueError:
                    pass
                continue
            if len(current_chunk) + len(para) > chunk_size and current_chunk:
                chunks.append({"content": current_chunk.strip(), "page_num": page_num})
                current_chunk = current_chunk[-overlap:] if overlap > 0 else ""
            current_chunk += "\n\n" + para if current_chunk else para
        if current_chunk.strip():
            chunks.append({"content": current_chunk.strip(), "page_num": page_num})
        return chunks
