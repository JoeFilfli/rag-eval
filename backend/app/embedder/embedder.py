import logging

from llama_index.core.schema import TextNode
from llama_index.vector_stores.postgres import PGVectorStore

from app.core.config import settings
from app.embedder.chunker import Chunk

logger = logging.getLogger(__name__)


def _get_embed_model():
    if settings.EMBEDDING_PROVIDER == "openai":
        from llama_index.embeddings.openai import OpenAIEmbedding
        return OpenAIEmbedding(model=settings.EMBEDDING_MODEL, api_key=settings.OPENAI_API_KEY)
    else:
        raise ValueError(f"Unknown EMBEDDING_PROVIDER: {settings.EMBEDDING_PROVIDER}")


def _get_vector_store() -> PGVectorStore:
    from urllib.parse import urlparse, unquote
    parsed = urlparse(settings.DATABASE_URL)
    return PGVectorStore.from_params(
        host=parsed.hostname,
        port=parsed.port or 5432,
        database=parsed.path.lstrip("/"),
        user=parsed.username,
        password=unquote(parsed.password or ""),
        table_name="chunks",
        embed_dim=settings.EMBEDDING_DIM,
    )


def embed_and_store(chunks: list[Chunk], document_name: str) -> int:
    """Embed each chunk and store in pgvector. Returns count of chunks stored."""
    if not chunks:
        return 0

    embed_model = _get_embed_model()
    vector_store = _get_vector_store()

    nodes: list[TextNode] = []
    for i, chunk in enumerate(chunks):
        try:
            embedding = embed_model.get_text_embedding(chunk.content)
            node = TextNode(
                text=chunk.content,
                embedding=embedding,
                metadata={
                    "page": chunk.page,
                    "type": chunk.type,
                    "document": chunk.document,
                },
            )
            nodes.append(node)
            logger.info(f"Embedded chunk {i + 1}/{len(chunks)} (page {chunk.page}, type={chunk.type})")
        except Exception as e:
            raise RuntimeError(f"Embedding failed at chunk {i + 1} of '{document_name}': {e}") from e

    vector_store.add(nodes)
    logger.info(f"Stored {len(nodes)} chunks for '{document_name}'")
    return len(nodes)
