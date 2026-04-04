import logging
from pathlib import Path

from gensim.models import KeyedVectors
from haystack import Pipeline, Document
from haystack.components.retrievers import InMemoryEmbeddingRetriever

from app.components import Passthrough, SimilarityEvaluator
from app.utils import get_or_create_document_store
from app.word2vec.components import Word2VecDocumentEmbedder, Word2VecTextEmbedder

logger = logging.getLogger("uvicorn.error")
cache = Path("./cache/word2vec.bin")


def create(model_path: Path, documents: list[Document]) -> Pipeline:
    if not model_path.exists():
        raise FileNotFoundError

    logger.info(f"Loading {model_path.name} Word2Vec model.")
    model = KeyedVectors.load_word2vec_format(str(model_path), binary=True)

    doc_embedder = Word2VecDocumentEmbedder(model=model)
    text_embedder = Word2VecTextEmbedder(model=model)
    evaluator = SimilarityEvaluator("valhalla/distilbart-mnli-12-3")

    logger.info("Creating Word2Vec document embeddings.")
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
