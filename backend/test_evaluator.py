# Quick test: run a full query + Ragas evaluation and store result in Postgres.
# Run from project root: .\backend\venv\Scripts\python .\backend\test_evaluator.py
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

from app.rag.pipeline import run_query
from app.evaluation.evaluator import evaluate_and_store

QUERY = "What is Apple's revenue in 2025?"

def main():
    print(f"\nQuery: {QUERY}\n")

    start = time.time()
    result = run_query(QUERY)
    latency_ms = int((time.time() - start) * 1000)

    print(f"Answer: {result.answer}")
    print(f"Latency: {latency_ms}ms")
    print(f"\nRunning Ragas evaluation...")

    evaluate_and_store(result, latency_ms)
    print("Done — row written to evaluations table.")

if __name__ == "__main__":
    main()
