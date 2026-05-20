import logging
from dataclasses import dataclass, field
from typing import Literal

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

ChunkType = Literal["text", "table", "image"]


@dataclass
class RawChunk:
    content: str
    page: int
    type: ChunkType
    document: str


@dataclass
class _ImageChunk(RawChunk):
    images_bytes: list[tuple[bytes, str]] = field(default_factory=list, repr=False)
    # Each tuple is (image_bytes, media_type) e.g. ("image/jpeg", "image/png")


def extract_raw_chunks(pdf_bytes: bytes, document_name: str) -> list[RawChunk]:
    """Extract text, tables, and images from PDF bytes. Skips pages that fail entirely."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    chunks: list[RawChunk] = []

    for page_num, page in enumerate(doc, start=1):
        try:
            page_chunks = _extract_page(page, page_num, document_name)
            chunks.extend(page_chunks)
        except Exception as e:
            logger.warning(f"Skipping page {page_num} of '{document_name}': {e}")

    doc.close()
    return chunks


def _extract_page(page: fitz.Page, page_num: int, document_name: str) -> list[RawChunk]:
    chunks: list[RawChunk] = []

    table_chunks, table_bboxes = _extract_tables(page, page_num, document_name)
    chunks.extend(table_chunks)

    text = _extract_text_excluding_tables(page, table_bboxes)
    if text:
        chunks.append(RawChunk(content=text, page=page_num, type="text", document=document_name))

    image_chunks = _extract_image_clips(page, page_num, document_name)
    chunks.extend(image_chunks)

    return chunks


def _extract_tables(page: fitz.Page, page_num: int, document_name: str) -> tuple[list[RawChunk], list[fitz.Rect]]:
    chunks: list[RawChunk] = []
    table_bboxes: list[fitz.Rect] = []
    try:
        finder = page.find_tables()
        for table in finder.tables:
            table_bboxes.append(fitz.Rect(table.bbox))
            markdown = _table_to_markdown(table.extract())
            if markdown:
                chunks.append(RawChunk(content=markdown, page=page_num, type="table", document=document_name))
    except Exception as e:
        logger.warning(f"Table extraction failed on page {page_num} of '{document_name}': {e}")
    return chunks, table_bboxes


def _extract_text_excluding_tables(page: fitz.Page, table_bboxes: list[fitz.Rect]) -> str:
    """Extract plain text blocks that do not overlap with any table bounding box."""
    blocks = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, block_type)
    text_parts: list[str] = []
    for block in blocks:
        block_rect = fitz.Rect(block[:4])
        block_text = block[4].strip()
        if not block_text:
            continue
        if any(block_rect.intersects(table_rect) for table_rect in table_bboxes):
            continue
        text_parts.append(block_text)
    return "\n".join(text_parts)


def _table_to_markdown(rows: list[list]) -> str:
    if not rows:
        return ""
    header = rows[0]
    body = rows[1:]
    md = "| " + " | ".join(str(cell or "") for cell in header) + " |"
    md += "\n| " + " | ".join("---" for _ in header) + " |"
    for row in body:
        md += "\n| " + " | ".join(str(cell or "") for cell in row) + " |"
    return md


def _extract_image_clips(page: fitz.Page, page_num: int, document_name: str) -> list[RawChunk]:
    """Collect all images on the page into a single _ImageChunk for one batched vision LLM call."""
    try:
        image_list = page.get_images(full=True)
        if not image_list:
            return []
        doc = page.parent
        all_images: list[tuple[bytes, str]] = []
        for img_info in image_list:
            xref = img_info[0]
            base_image = doc.extract_image(xref)
            ext = base_image.get("ext", "png").lower()
            media_type = f"image/{ext}" if ext != "jpg" else "image/jpeg"
            all_images.append((base_image["image"], media_type))
        return [_ImageChunk(content="", page=page_num, type="image", document=document_name, images_bytes=all_images)]
    except Exception as e:
        logger.warning(f"Image extraction failed on page {page_num} of '{document_name}': {e}")
        return []
