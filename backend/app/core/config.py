import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    DOCUMENT_STORE_PATH: str = os.getenv("DOCUMENT_STORE_PATH", "./documents")

    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o")

    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "openai")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")

    EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIM", "1536"))

    RETRIEVAL_K: int = int(os.getenv("RETRIEVAL_K", "5"))
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "512"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))

    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/rag")


settings = Settings()
