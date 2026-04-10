import logging
from pathlib import Path

from haystack import Document
from sentence_transformers import SentenceTransformer

from app.components import SentenceTransformerQueryEmbedder, SentenceTransformerDocumentEmbedder
from app.utils import get_or_create_document_store, PipelineFactory

logger = logging.getLogger("uvicorn.error")
cache = Path("./cache/llm.bin")
model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device=None)


def create(documents: list[Document], *, force_embeddings: bool = False) -> PipelineFactory:
    query_embedder = SentenceTransformerQueryEmbedder(model)
    doc_embedder = SentenceTransformerDocumentEmbedder(model)

    logger.info("Creating LLM embeddings.")
    document_store = get_or_create_document_store(
        documents, embedder=doc_embedder, cache_path=cache, force_refresh=force_embeddings
    )

    return PipelineFactory(query_embedder, doc_embedder, document_store)


__all__ = (
    "create",
)
