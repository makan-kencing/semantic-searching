import logging
from pathlib import Path

from haystack import Pipeline, Document
from haystack.components.retrievers import InMemoryEmbeddingRetriever

from app.bert.components import BERTDocumentEmbedder, BERTTextEmbedder
from app.components import Passthrough, SimilarityEvaluator
from app.utils import get_or_create_document_store

logger = logging.getLogger("uvicorn.error")
cache = Path("./cache/bert.bin")


def create(documents: list[Document]) -> Pipeline:
    doc_embedder = BERTDocumentEmbedder()
    text_embedder = BERTTextEmbedder()
    evaluator = SimilarityEvaluator("valhalla/distilbart-mnli-12-3")

    logger.info("Creating BERT embeddings.")
    document_store = get_or_create_document_store(documents, embedder=doc_embedder, cache_path=cache)

    pipeline = Pipeline()
    pipeline.add_component(name="query", instance=Passthrough(query=str, top_k=int))
    pipeline.add_component(name="embedder", instance=text_embedder)
    pipeline.add_component(name="retriever", instance=InMemoryEmbeddingRetriever(document_store=document_store))
    pipeline.add_component(name="evaluator", instance=evaluator)
    pipeline.add_component(name="result", instance=Passthrough(documents=list[Document], similarity=list[float]))

    pipeline.connect("query.query", "embedder.text")
    pipeline.connect("query.top_k", "retriever.top_k")
    pipeline.connect("embedder.embedding", "retriever.query_embedding")
    pipeline.connect("query.query", "evaluator.query")
    pipeline.connect("retriever.documents", "evaluator.documents")
    pipeline.connect("retriever.documents", "result.documents")
    pipeline.connect("evaluator.scores", "result.similarity")

    return pipeline


__all__ = (
    "create",
)
