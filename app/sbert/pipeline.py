import logging
from pathlib import Path

from haystack import Pipeline, Document
from haystack.components.builders import PromptBuilder
from haystack.components.converters import OutputAdapter
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.generators import HuggingFaceLocalGenerator
from haystack.components.joiners import DocumentJoiner
from haystack.components.retrievers import InMemoryBM25Retriever, InMemoryEmbeddingRetriever

from app.components import Passthrough, HypotheticalDocumentEmbedder, SimilarityEvaluator
from app.utils import get_or_create_document_store

logger = logging.getLogger("uvicorn.error")
cache = Path("./cache/sbert.bin")


def create(documents: list[Document]) -> Pipeline:
    generator = HuggingFaceLocalGenerator(
        model="google/flan-t5-large",
        task="text2text-generation",
        generation_kwargs={
            "max_new_tokens": 100,
            "temperature": 0.9,
        },
    )

    prompt_builder = PromptBuilder(
        template="""Given a question, generate a paragraph of text that answers the question.    Question: {{question}}    Paragraph:""",
    )

    document_adapter = OutputAdapter(
        template="{{ answers | build_doc }}",
        output_type=list[Document],
        custom_filters={"build_doc": lambda data: [Document(content=d) for d in data]},
        unsafe=True
    )

    embedder = SentenceTransformersDocumentEmbedder(
        model="sentence-transformers/all-MiniLM-L6-v2",
    )
    embedder.warm_up()

    evaluator = SimilarityEvaluator("valhalla/distilbart-mnli-12-3")

    logger.info("Creating sentence transformers embeddings.")
    document_store = get_or_create_document_store(documents, embedder=embedder, cache_path=cache)

    pipeline = Pipeline()
    pipeline.add_component(name="query", instance=Passthrough(query=str, top_k=int))
    pipeline.add_component(name="prompt_builder", instance=prompt_builder)
    pipeline.add_component(name="generator", instance=generator)
    pipeline.add_component(name="document_adapter", instance=document_adapter)
    pipeline.add_component(name="embedder", instance=embedder)
    pipeline.add_component(name="hyde", instance=HypotheticalDocumentEmbedder())
    pipeline.add_component(name="bm25_retriever", instance=InMemoryBM25Retriever(document_store=document_store))
    pipeline.add_component(name="embedding_retriever",
                           instance=InMemoryEmbeddingRetriever(document_store=document_store))
    pipeline.add_component(name="joiner", instance=DocumentJoiner(join_mode="reciprocal_rank_fusion", top_k=10))
    pipeline.add_component(name="evaluator", instance=evaluator)
    pipeline.add_component(name="result", instance=Passthrough(documents=list[Document], similarity=list[float]))

    pipeline.connect("query.query", "prompt_builder.question")
    pipeline.connect("prompt_builder", "generator")
    pipeline.connect("generator.replies", "document_adapter.answers")
    pipeline.connect("document_adapter.output", "embedder.documents")
    pipeline.connect("embedder.documents", "hyde.documents")
    pipeline.connect("query.top_k", "embedding_retriever.top_k")
    pipeline.connect("hyde.hypothetical_embedding", "embedding_retriever.query_embedding")
    pipeline.connect("query.top_k", "bm25_retriever.top_k")
    pipeline.connect("query.query", "bm25_retriever.query")
    pipeline.connect("query.top_k", "joiner.top_k")
    pipeline.connect("embedding_retriever.documents", "joiner.documents")
    pipeline.connect("bm25_retriever.documents", "joiner.documents")
    pipeline.connect("query.query", "evaluator.query")
    pipeline.connect("joiner.documents", "evaluator.documents")
    pipeline.connect("joiner.documents", "result.documents")
    pipeline.connect("evaluator.scores", "result.similarity")

    return pipeline


__all__ = (
    "create",
)
