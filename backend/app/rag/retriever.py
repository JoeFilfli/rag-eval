from dataclasses import dataclass
from urllib.parse import urlparse, unquote

from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.vector_stores.postgres import PGVectorStore

from app.core.config import settings
from app.embedder.embedder import _get_embed_model


@dataclass
class RetrievedChunk:
    content: str
    page: int
    type: str
    document: str
    score: float


def retrieve(query: str, k: int) -> list[RetrievedChunk]:
    """Embed the query and return top-k chunks from pgvector by cosine similarity."""
    embed_model = _get_embed_model()
    query_embedding = embed_model.get_text_embedding(query)

    parsed = urlparse(settings.DATABASE_URL)
    vector_store = PGVectorStore.from_params(
        host=parsed.hostname,
        port=parsed.port or 5432,
        database=parsed.path.lstrip("/"),
        user=parsed.username,
        password=unquote(parsed.password or ""),
        table_name="chunks",
        embed_dim=settings.EMBEDDING_DIM,
    )

    results = vector_store.query(
        VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=k,
        )
    )

    chunks: list[RetrievedChunk] = []
    for node, score in zip(results.nodes, results.similarities or []):
        meta = node.metadata
        chunks.append(RetrievedChunk(
            content=node.text,
            page=int(meta.get("page", 0)),
            type=meta.get("type", "text"),
            document=meta.get("document", ""),
            score=score or 0.0,
        ))
    return chunks
