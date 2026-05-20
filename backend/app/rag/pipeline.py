import json
import logging
import time
from dataclasses import dataclass, field

from app.core.config import settings
from app.core.llm import get_llm_client_with_usage
from app.rag.judge import judge_chunks, rephrase_query
from app.rag.retriever import RetrievedChunk, retrieve

logger = logging.getLogger(__name__)

GENERATE_PROMPT = """Answer the question using only the information in the provided context chunks.

Rules:
- Only use facts explicitly stated in the context.
- If the context is about a different company, product, or entity than what was asked, do not use it — say the documents do not contain information about the requested subject.
- Do not invent facts, infer from similar data, or substitute one entity for another.
- If the context is insufficient to answer fully, say so explicitly.

Question: {query}

Context:
{context}

Answer:"""


@dataclass
class QueryResult:
    answer: str
    approved_chunks: list[RetrievedChunk]
    low_confidence: bool
    trace: dict = field(default_factory=dict)


def run_query(query: str) -> QueryResult:
    k = settings.RETRIEVAL_K
    rephrased_query = None

    # Step 1: retrieve
    chunks = retrieve(query, k)
    logger.info(f"Retrieved {len(chunks)} chunks for query: {query!r}")

    # Step 2: judge
    approved, sufficient = judge_chunks(query, chunks)
    logger.info(f"Approved {len(approved)}/{len(chunks)} chunks — sufficient={sufficient}")

    # Step 3: re-retrieve once if insufficient
    if not sufficient:
        rephrased_query = rephrase_query(query)
        k_doubled = k * 2
        logger.info(f"Re-retrieving with k={k_doubled}, rephrased query: {rephrased_query!r}")
        chunks2 = retrieve(rephrased_query, k_doubled)
        approved, sufficient = judge_chunks(rephrased_query, chunks2)
        logger.info(f"After re-retrieval: approved {len(approved)}/{len(chunks2)} — sufficient={sufficient}")

    low_confidence = not sufficient or not approved

    # Step 4: generate
    if approved:
        context = "\n\n".join(
            f"[Page {c.page}, {c.type}]\n{c.content}" for c in approved
        )
        answer_prompt = GENERATE_PROMPT.format(query=query, context=context)
        chat = get_llm_client_with_usage()
        t0 = time.time()
        answer, usage = chat([{"role": "user", "content": answer_prompt}])
        print(json.dumps({
            "activity": "generate",
            "latency_ms": int((time.time() - t0) * 1000),
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
        }), flush=True)
    else:
        answer = "I could not find sufficient context in the uploaded documents to answer this question."

    trace = {
        "query": query,
        "rephrased_query": rephrased_query,
        "k_initial": settings.RETRIEVAL_K,
        "k_final": k * 2 if rephrased_query else k,
        "chunks_retrieved": len(chunks),
        "chunks_approved": len(approved),
    }

    return QueryResult(
        answer=answer,
        approved_chunks=approved,
        low_confidence=low_confidence,
        trace=trace,
    )
