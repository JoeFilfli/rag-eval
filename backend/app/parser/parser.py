import logging

from app.parser.extractor import RawChunk, _ImageChunk, extract_raw_chunks
from app.parser.loader import load_pdf_bytes
from app.parser.vision import describe_images

logger = logging.getLogger(__name__)


def parse_document(file_path: str, document_name: str) -> list[RawChunk]:
    """
    Full parse pipeline: load PDF → extract text/tables/images → describe images via vision LLM.
    Returns a flat list of RawChunks. Pages that fail entirely are skipped with a warning.
    """
    pdf_bytes = load_pdf_bytes(file_path)
    raw_chunks = extract_raw_chunks(pdf_bytes, document_name)

    final_chunks: list[RawChunk] = []
    pages_skipped = 0

    for chunk in raw_chunks:
        if isinstance(chunk, _ImageChunk):
            description = describe_images(chunk.images_bytes)
            if description is None:
                logger.warning(f"Skipping images on page {chunk.page} of '{document_name}' — vision LLM returned nothing")
                pages_skipped += 1
                continue
            chunk.content = description

        if chunk.content.strip():
            final_chunks.append(chunk)

    logger.info(
        f"Parsed '{document_name}': {len(final_chunks)} chunks produced, {pages_skipped} image(s) skipped"
    )
    return final_chunks
