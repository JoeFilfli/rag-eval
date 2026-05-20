# Quick test: chunk and embed first 3 pages of the 2025 annual report into pgvector.
# Run from backend/ with: venv\Scripts\python test_embedder.py
import sys
from pathlib import Path

import fitz

sys.path.insert(0, str(Path(__file__).parent))
from app.parser.extractor import extract_raw_chunks, _ImageChunk
from app.parser.vision import describe_images
from app.embedder.chunker import split_chunks
from app.embedder.embedder import embed_and_store

PDF_PATH = Path(__file__).parent.parent / "reports" / "2025_annual_report_20260410152613.pdf"
MAX_PAGES = 5


def main():
    print(f"Opening: {PDF_PATH.name} (first {MAX_PAGES} pages)")
    doc = fitz.open(str(PDF_PATH))
    sub_doc = fitz.open()
    sub_doc.insert_pdf(doc, from_page=0, to_page=MAX_PAGES - 1)
    pdf_bytes = sub_doc.tobytes()
    doc.close()
    sub_doc.close()

    print("Step 1: Extracting chunks...")
    raw_chunks = extract_raw_chunks(pdf_bytes, PDF_PATH.name)

    print(f"Step 2: Describing {sum(1 for c in raw_chunks if c.type == 'image')} image chunk(s) via vision LLM...")
    for chunk in raw_chunks:
        if isinstance(chunk, _ImageChunk):
            description = describe_images(chunk.images_bytes)
            chunk.content = description or "[vision failed]"

    print("Step 3: Splitting text/table chunks...")
    chunks = split_chunks(raw_chunks)
    by_type = {"text": 0, "table": 0, "image": 0}
    for c in chunks:
        by_type[c.type] += 1
    print(f"  After splitting: {len(chunks)} chunks — text={by_type['text']} table={by_type['table']} image={by_type['image']}")

    print("Step 4: Embedding and storing in pgvector...")
    stored = embed_and_store(chunks, PDF_PATH.name)
    print(f"\nDone. {stored} chunks stored in pgvector.")


if __name__ == "__main__":
    main()
