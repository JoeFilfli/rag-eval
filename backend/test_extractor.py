# Quick test: extract first 20 pages of the 2025 annual report and save chunks to a file.
# Run from backend/ with: venv\Scripts\python test_extractor.py
import json
import os
import sys
from pathlib import Path

os.environ["LLM_PROVIDER"] = "anthropic"
os.environ["LLM_MODEL"] = "claude-haiku-4-5-20251001"
os.environ["ANTHROPIC_API_KEY"] = os.environ.get("ANTHROPIC_API_KEY", "")

import fitz

sys.path.insert(0, str(Path(__file__).parent))
from app.parser.extractor import extract_raw_chunks, _ImageChunk
from app.parser.vision import describe_images

PDF_PATH = Path(__file__).parent.parent / "reports" / "2025_annual_report_20260410152613.pdf"
OUTPUT_PATH = Path(__file__).parent.parent / "reports" / "extractor_test_output.json"
MAX_PAGES = 20


def main():
    print(f"Opening: {PDF_PATH.name}")
    doc = fitz.open(str(PDF_PATH))
    total_pages = len(doc)
    print(f"Total pages: {total_pages} — extracting first {min(MAX_PAGES, total_pages)}")

    sub_doc = fitz.open()
    sub_doc.insert_pdf(doc, from_page=0, to_page=min(MAX_PAGES, total_pages) - 1)
    pdf_bytes = sub_doc.tobytes()
    doc.close()
    sub_doc.close()

    raw_chunks = extract_raw_chunks(pdf_bytes, PDF_PATH.name)

    output = []
    by_type = {"text": 0, "table": 0, "image": 0}

    for i, chunk in enumerate(raw_chunks):
        if isinstance(chunk, _ImageChunk):
            print(f"  Describing {len(chunk.images_bytes)} image(s) on page {chunk.page} ({', '.join(mt for _, mt in chunk.images_bytes)})...")
            description = describe_images(chunk.images_bytes)
            content = description if description else "[vision LLM returned nothing]"
        else:
            content = chunk.content

        by_type[chunk.type] += 1
        output.append({
            "page": chunk.page,
            "type": chunk.type,
            "document": chunk.document,
            "content": content,
        })

    OUTPUT_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nChunks produced: {len(output)}")
    print(f"  text:  {by_type['text']}")
    print(f"  table: {by_type['table']}")
    print(f"  image: {by_type['image']}")
    print(f"\nOutput saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
