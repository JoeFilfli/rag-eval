import json
import logging
import time

from app.core.llm import get_llm_client_with_usage
from app.rag.retriever import RetrievedChunk

logger = logging.getLogger(__name__)

JUDGE_PROMPT = """You are evaluating retrieved chunks for a RAG system.

Query: {query}

Retrieved chunks:
{chunks}

For each chunk, score its marginal contribution to answering the query given all other chunks:
- "relevant": adds unique, necessary information that directly addresses the query — the subject, entity, and topic must match
- "irrelevant": does not address the query, is off-topic, or is about a different entity or company than what was asked
- "insufficient": topically related and about the right subject, but lacks enough detail on its own — consider the full set

A chunk about a different company, product, or entity than the one asked about must be scored "irrelevant", even if it contains similar types of data.

Also determine if the OVERALL set of relevant chunks is sufficient to answer the query fully. If no relevant chunks remain, the overall must be "insufficient".

Respond with valid JSON only:
{{
  "scores": [{{"index": 0, "score": "relevant"}}, ...],
  "overall": "sufficient" | "insufficient"
}}"""

REPHRASE_PROMPT = """The following query did not retrieve sufficient context. Rephrase it to target different or more specific information.

Original query: {query}

Return only the rephrased query, nothing else."""


def judge_chunks(query: str, chunks: list[RetrievedChunk]) -> tuple[list[RetrievedChunk], bool]:
    """
    Score each chunk's marginal contribution. Return (approved_chunks, is_sufficient).
    """
    chat = get_llm_client_with_usage()

    chunks_text = "\n\n".join(
        f"[{i}] (page {c.page}, type={c.type})\n{c.content}"
        for i, c in enumerate(chunks)
    )

    prompt = JUDGE_PROMPT.format(query=query, chunks=chunks_text)
    t0 = time.time()
    response, usage = chat([{"role": "user", "content": prompt}])
    print(json.dumps({
        "activity": "judge",
        "latency_ms": int((time.time() - t0) * 1000),
        "input_tokens": usage["input_tokens"],
        "output_tokens": usage["output_tokens"],
    }), flush=True)

    try:
        # Strip markdown code fences if present
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        parsed = json.loads(cleaned.strip())
        scores = {item["index"]: item["score"] for item in parsed["scores"]}
        overall = parsed.get("overall", "insufficient")
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Judge response parse failed: {e} — keeping all chunks")
        return chunks, True

    approved = [c for i, c in enumerate(chunks) if scores.get(i) != "irrelevant"]
    is_sufficient = overall == "sufficient" and bool(approved)
    return approved, is_sufficient


def rephrase_query(query: str) -> str:
    """Ask the LLM to rephrase the query to retrieve different context."""
    from app.core.llm import get_llm_client
    chat = get_llm_client()
    response = chat([{"role": "user", "content": REPHRASE_PROMPT.format(query=query)}])
    return response.strip()
