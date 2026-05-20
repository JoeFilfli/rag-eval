from dataclasses import dataclass
from typing import Literal

from llama_index.core.node_parser import SentenceSplitter

from app.core.config import settings
from app.parser.extractor import RawChunk

ChunkType = Literal["text", "table", "image"]


@dataclass
class Chunk:
    content: str
    page: int
    type: ChunkType
    document: str


def split_chunks(raw_chunks: list[RawChunk]) -> list[Chunk]:
    """
    Split text and table chunks into fixed-size pieces using SentenceSplitter.
    Image description chunks pass through unsplit.
    """
    splitter = SentenceSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )
    result: list[Chunk] = []

    for raw in raw_chunks:
        if raw.type == "image":
            result.append(Chunk(content=raw.content, page=raw.page, type=raw.type, document=raw.document))
            continue

        pieces = splitter.split_text(raw.content)
        for piece in pieces:
            if piece.strip():
                result.append(Chunk(content=piece.strip(), page=raw.page, type=raw.type, document=raw.document))

    return result
