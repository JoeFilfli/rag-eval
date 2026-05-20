# Quick test: run a query against the 3 embedded pages.
# Run from project root: .\backend\venv\Scripts\python .\backend\test_pipeline.py
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

from app.rag.pipeline import run_query

QUERY = "What is Apple's revenue in 2025?"

def main():
    print(f"\nQuery: {QUERY}\n")
    result = run_query(QUERY)
    print(f"Answer:\n{result.answer}\n")
    print(f"Low confidence: {result.low_confidence}")
    print(f"Approved chunks: {result.trace['chunks_approved']}/{result.trace['chunks_retrieved']}")
    print(f"Rephrased query: {result.trace['rephrased_query']}")
    print(f"\nSources:")
    for c in result.approved_chunks:
        print(f"  - page {c.page} [{c.type}] {c.document}")

if __name__ == "__main__":
    main()
