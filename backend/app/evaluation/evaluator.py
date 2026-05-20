import json
import logging
import time
import psycopg2

from app.core.config import settings
from app.rag.pipeline import QueryResult

logger = logging.getLogger(__name__)


def _get_ragas_llm():
    if settings.LLM_PROVIDER == "openai":
        from ragas.llms import LangchainLLMWrapper
        from langchain_openai import ChatOpenAI
        return LangchainLLMWrapper(ChatOpenAI(model=settings.LLM_MODEL, api_key=settings.OPENAI_API_KEY))
    elif settings.LLM_PROVIDER == "anthropic":
        from ragas.llms import LangchainLLMWrapper
        from langchain_anthropic import ChatAnthropic
        return LangchainLLMWrapper(ChatAnthropic(model=settings.LLM_MODEL, api_key=settings.ANTHROPIC_API_KEY))
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {settings.LLM_PROVIDER}")


def _get_ragas_embeddings():
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_openai import OpenAIEmbeddings
    return LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=settings.EMBEDDING_MODEL, api_key=settings.OPENAI_API_KEY))


def _run_ragas(query: str, answer: str, contexts: list[str]) -> dict:
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_recall
    from datasets import Dataset

    llm = _get_ragas_llm()
    embeddings = _get_ragas_embeddings()

    dataset = Dataset.from_dict({
        "question": [query],
        "answer": [answer],
        "contexts": [contexts],
        "ground_truth": [answer],  # using answer as ground truth since we have no labelled set
    })

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_recall],
        llm=llm,
        embeddings=embeddings,
    )
    return result


def _write_to_db(
    query: str,
    answer: str,
    low_confidence: bool,
    faithfulness: float | None,
    answer_relevancy: float | None,
    context_recall: float | None,
    latency_ms: int,
    approved_chunks: list,
    trace: dict,
):
    conn = psycopg2.connect(settings.DATABASE_URL)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO evaluations
                   (query, answer, low_confidence, faithfulness, answer_relevancy,
                    context_recall, latency_ms, approved_chunks, trace)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (
                    query, answer, low_confidence,
                    faithfulness, answer_relevancy, context_recall,
                    latency_ms,
                    json.dumps([
                        {"page": c.page, "type": c.type, "document": c.document, "content": c.content}
                        for c in approved_chunks
                    ]),
                    json.dumps(trace),
                ),
            )
        conn.commit()
    finally:
        conn.close()


def evaluate_and_store(result: QueryResult, latency_ms: int) -> dict:
    """Run Ragas metrics on the query result, write one row to the evaluations table, and return the scores."""
    contexts = [c.content for c in result.approved_chunks]
    faithfulness_score = answer_relevancy_score = context_recall_score = None
    ragas_failed = False

    t0 = time.time()
    try:
        scores = _run_ragas(result.trace["query"], result.answer, contexts)

        def _to_float(val) -> float:
            if isinstance(val, list):
                return float(sum(val) / len(val)) if val else 0.0
            return float(val)

        faithfulness_score = _to_float(scores["faithfulness"])
        answer_relevancy_score = _to_float(scores["answer_relevancy"])
        context_recall_score = _to_float(scores["context_recall"])
        logger.info(
            f"Ragas scores — faithfulness={faithfulness_score:.2f} "
            f"answer_relevancy={answer_relevancy_score:.2f} "
            f"context_recall={context_recall_score:.2f}"
        )
    except Exception as e:
        ragas_failed = True
        logger.error(f"Ragas evaluation failed: {e} — storing row with null scores")
    finally:
        print(json.dumps({
            "activity": "evaluate",
            "latency_ms": int((time.time() - t0) * 1000),
            "ragas_failed": ragas_failed,
            "low_confidence": result.low_confidence,
        }), flush=True)

    _write_to_db(
        query=result.trace["query"],
        answer=result.answer,
        low_confidence=result.low_confidence,
        faithfulness=faithfulness_score,
        answer_relevancy=answer_relevancy_score,
        context_recall=context_recall_score,
        latency_ms=latency_ms,
        approved_chunks=result.approved_chunks,
        trace=result.trace,
    )
    logger.info(f"Evaluation stored — latency={latency_ms}ms")
    return {
        "faithfulness": faithfulness_score,
        "answer_relevancy": answer_relevancy_score,
        "context_recall": context_recall_score,
    }
