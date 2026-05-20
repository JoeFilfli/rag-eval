from pathlib import Path

from app.core.config import settings


def load_pdf_bytes(file_path: str) -> bytes:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {file_path}")
    return path.read_bytes()


def save_pdf_bytes(filename: str, data: bytes) -> str:
    folder = Path(settings.DOCUMENT_STORE_PATH)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / filename
    path.write_bytes(data)
    return str(path)
