import json
import logging
import math
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import psycopg2
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.core.config import settings
from app.embedder.chunker import split_chunks
from app.embedder.embedder import embed_and_store
from app.evaluation.evaluator import evaluate_and_store
from app.parser.loader import save_pdf_bytes
from app.parser.parser import parse_document
from app.rag.pipeline import run_query

logger = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="RAG Pipeline")


@app.on_event("startup")
def _init_db():
    conn = _db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS evaluations (
                    id            SERIAL PRIMARY KEY,
                    query         TEXT,
                    answer        TEXT,
                    low_confidence BOOLEAN,
                    faithfulness  FLOAT,
                    answer_relevancy FLOAT,
                    context_recall FLOAT,
                    latency_ms    INTEGER,
                    approved_chunks JSONB,
                    trace         JSONB,
                    created_at    TIMESTAMP DEFAULT NOW()
                )
            """)
        conn.commit()
    finally:
        conn.close()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
_eval_executor = ThreadPoolExecutor(max_workers=2)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str
    run_evaluation: bool = False


def _db_conn():
    return psycopg2.connect(settings.DATABASE_URL)


def _safe_float(v):
    if v is None:
        return None
    try:
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return None


@app.post("/api/documents")
@limiter.limit("5/hour")
async def upload_document(request: Request, file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    try:
        data = await file.read()
        filename = f"{uuid.uuid4()}_{file.filename}"
        file_path = save_pdf_bytes(filename, data)
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": str(e), "step": "upload"})

    try:
        raw_chunks = parse_document(file_path, file.filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": str(e), "step": "parse"})

    pages_skipped = sum(1 for c in raw_chunks if not c.content.strip())

    try:
        chunks = split_chunks(raw_chunks)
        chunk_count = embed_and_store(chunks, file.filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": str(e), "step": "embed"})

    return {
        "document_id": filename,
        "chunk_count": chunk_count,
        "pages_skipped": pages_skipped,
    }


@app.post("/api/query")
@limiter.limit("10/minute")
async def query(request: Request, body: QueryRequest):
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        start = time.time()
        result = run_query(body.question)
        latency_ms = int((time.time() - start) * 1000)
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": str(e), "step": "query"})

    scores = {"faithfulness": None, "answer_relevancy": None, "context_recall": None}

    if body.run_evaluation:
        try:
            scores = evaluate_and_store(result, latency_ms)
        except Exception as e:
            logger.error(f"Evaluation failed but query succeeded: {e}")
    else:
        def _run_eval():
            try:
                evaluate_and_store(result, latency_ms)
            except Exception as e:
                logger.error(f"Background evaluation failed: {e}")
        _eval_executor.submit(_run_eval)

    return {
        "answer": result.answer,
        "low_confidence": result.low_confidence,
        "latency_ms": latency_ms,
        "faithfulness": scores["faithfulness"],
        "answer_relevancy": scores["answer_relevancy"],
        "context_recall": scores["context_recall"],
        "sources": [
            {"page": c.page, "type": c.type, "document": c.document}
            for c in result.approved_chunks
        ],
    }


@app.get("/api/documents")
async def list_documents():
    try:
        conn = _db_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT
                        metadata_->>'document' AS document,
                        COUNT(*) AS chunk_count
                    FROM data_chunks
                    WHERE metadata_->>'document' IS NOT NULL
                    GROUP BY metadata_->>'document'
                    ORDER BY document
                """)
                rows = cur.fetchall()
        finally:
            conn.close()
    except psycopg2.errors.UndefinedTable:
        return []
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": str(e), "step": "documents"})

    return [{"document": row[0], "chunk_count": row[1]} for row in rows]


@app.delete("/api/documents/{document_name:path}")
async def delete_document(document_name: str):
    try:
        conn = _db_conn()
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM data_chunks WHERE metadata_->>'document' = %s",
                (document_name,),
            )
            deleted = cur.rowcount
        conn.commit()
        conn.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": str(e), "step": "delete"})

    return {"document": document_name, "chunks_deleted": deleted}


@app.get("/api/history")
async def history():
    try:
        conn = _db_conn()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT query, answer, low_confidence, faithfulness, answer_relevancy,
                       context_recall, latency_ms, created_at
                FROM evaluations
                ORDER BY created_at DESC
                LIMIT 50
            """)
            rows = cur.fetchall()
        conn.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": str(e), "step": "history"})

    return [
        {
            "query": row[0],
            "answer": row[1],
            "low_confidence": row[2],
            "faithfulness": _safe_float(row[3]),
            "answer_relevancy": _safe_float(row[4]),
            "context_recall": _safe_float(row[5]),
            "latency_ms": row[6],
            "created_at": row[7].isoformat() if row[7] else None,
        }
        for row in rows
    ]
