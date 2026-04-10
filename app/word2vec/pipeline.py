import logging
from pathlib import Path

import gensim.downloader as api
from haystack import Document

from app.utils import get_or_create_document_store, PipelineFactory
from app.word2vec.components import Word2VecDocumentEmbedder, Word2VecTextEmbedder

logger = logging.getLogger("uvicorn.error")
cache = Path("./cache/word2vec.bin")
model = api.load("word2vec-google-news-300")


def create(documents: list[Document], *, force_embeddings: bool = False) -> PipelineFactory:
    logger.info("Loading Word2Vec embedder.")
    doc_embedder = Word2VecDocumentEmbedder(model)
    text_embedder = Word2VecTextEmbedder(model)

    logger.info("Creating Word2Vec document embeddings.")
    document_store = get_or_create_document_store(
        documents, embedder=doc_embedder, cache_path=cache, force_refresh=force_embeddings
    )

    return PipelineFactory(text_embedder, doc_embedder, document_store)


__all__ = (
    "create",
)
