import logging
from pathlib import Path

from haystack import Document

from app.utils import get_or_create_document_store, PipelineFactory
from app.word2vec.components import Word2VecDocumentEmbedder, Word2VecTextEmbedder

logger = logging.getLogger("uvicorn.error")
cache = Path("./cache/word2vec.bin")


def create(documents: list[Document], *, force_embeddings: bool = False) -> PipelineFactory:
    logger.info("Loading Word2Vec embedder.")
    doc_embedder = Word2VecDocumentEmbedder(model="word2vec-google-news-300")
    text_embedder = Word2VecTextEmbedder(model="word2vec-google-news-300")

    logger.info("Creating Word2Vec document embeddings.")
    document_store = get_or_create_document_store(documents, embedder=doc_embedder, cache_path=cache, force_refresh=force_embeddings)

    return PipelineFactory(text_embedder, doc_embedder, document_store)


__all__ = (
    "create",
)
