import logging
from pathlib import Path

from haystack import Document
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder

from app.utils import get_or_create_document_store, PipelineFactory

logger = logging.getLogger("uvicorn.error")
cache = Path("./cache/sbert.bin")


def create(documents: list[Document], *, force_embeddings: bool = False) -> PipelineFactory:
    logger.info("Loading sentence transformers embedder.")
    text_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
    text_embedder.warm_up()

    doc_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
    doc_embedder.warm_up()

    logger.info("Creating sentence transformers document embeddings.")
    document_store = get_or_create_document_store(documents, embedder=doc_embedder, cache_path=cache, force_refresh=force_embeddings)

    return PipelineFactory(text_embedder, doc_embedder, document_store)


__all__ = (
    "create",
)
