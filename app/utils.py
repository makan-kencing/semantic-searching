from pathlib import Path
from typing import Self

from haystack import Document, Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.converters import OutputAdapter
from haystack.components.generators import HuggingFaceLocalGenerator
from haystack.components.joiners import DocumentJoiner
from haystack.components.retrievers import InMemoryBM25Retriever, InMemoryEmbeddingRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from txtai.pipeline import Similarity

from app.components import SimilarityEvaluator, HypotheticalDocumentEmbedder, Passthrough


def get_or_create_document_store(
        documents: list[Document],
        *,
        embedder,
        cache_path: Path,
        force_refresh: bool = False
) -> InMemoryDocumentStore:
    cache_path.parent.mkdir(exist_ok=True, parents=True)

    if not cache_path.exists() or force_refresh:
        document_store = InMemoryDocumentStore(embedding_similarity_function="cosine")
        document_with_embeddings = embedder.run(documents)["documents"]
        document_store.write_documents(document_with_embeddings)
        document_store.save_to_disk(str(cache_path))
        return document_store
    return InMemoryDocumentStore.load_from_disk(str(cache_path))


similarity = Similarity("valhalla/distilbart-mnli-12-3")

class PipelineFactory:
    def __init__(self, text_embedder, document_embedder, document_store):
        self._text_embedder = text_embedder
        self._document_embedder = document_embedder
        self._document_store = document_store

        self._with_evaluator = False
        self._with_bm25 = False
        self._with_hyde = False

    def with_hyde(self) -> Self:
        self._with_hyde = True
        return self

    def with_bm25(self) -> Self:
        self._with_bm25 = True
        return self

    def with_evaluator(self) -> Self:
        self._with_evaluator = True
        return self

    def _use_hyde_embeddings(self, pipeline: Pipeline, sender: str, receiver: str) -> None:
        prompt_builder = PromptBuilder(
            template="""Given a question, generate a paragraph of text that answers the question.    Question: {{question}}    Paragraph:""",
        )

        generator = HuggingFaceLocalGenerator(
            model="google/flan-t5-large",
            task="text2text-generation",
            generation_kwargs={
                "max_new_tokens": 100,
                "temperature": 0.9,
            },
        )

        document_adapter = OutputAdapter(
            template="{{ answers | build_doc }}",
            output_type=list[Document],
            custom_filters={"build_doc": lambda data: [Document(content=d) for d in data]},
            unsafe=True
        )

        hyde = HypotheticalDocumentEmbedder()

        pipeline.add_component(name="prompt_builder", instance=prompt_builder)  # noqa
        pipeline.add_component(name="generator", instance=generator)  # noqa
        pipeline.add_component(name="document_adapter", instance=document_adapter)  # noqa
        pipeline.add_component(name="document_embedder", instance=self._document_embedder)
        pipeline.add_component(name="hyde", instance=hyde)  # noqa

        pipeline.connect(sender, "prompt_builder.question")
        pipeline.connect("prompt_builder", "generator")
        pipeline.connect("generator.replies", "document_adapter.answers")
        pipeline.connect("document_adapter.output", "document_embedder.documents")
        pipeline.connect("document_embedder.documents", "hyde.documents")
        pipeline.connect("hyde.hypothetical_embedding", receiver)

    def _use_query_embeddings(self, pipeline: Pipeline, sender: str, receiver: str) -> None:
        pipeline.add_component(name="embedder", instance=self._text_embedder)  # noqa

        pipeline.connect(sender, "embedder.text")
        pipeline.connect("embedder.embedding", receiver)

    def _add_bm25(self, pipeline: Pipeline, sender: str, receiver: str, *, sender_top_k: str = None) -> None:
        pipeline.add_component(name="bm25_retriever", instance=InMemoryBM25Retriever(document_store=self._document_store))  # noqa

        if sender_top_k is not None:
            pipeline.connect(sender_top_k, "bm25_retriever.top_k")
        pipeline.connect(sender, "bm25_retriever.query")
        pipeline.connect("bm25_retriever.documents", receiver)

    def _add_evaluator(self, pipeline: Pipeline, sender_query: str, sender_documents: str, receiver: str = None) -> None:
        evaluator = SimilarityEvaluator(similarity)

        pipeline.add_component(name="evaluator", instance=evaluator)  # noqa

        pipeline.connect(sender_query, "evaluator.query")
        pipeline.connect(sender_documents, "evaluator.documents")
        if receiver is not None:
            pipeline.connect("evaluator.scores", receiver)

    def make(self) -> Pipeline:
        pipeline = Pipeline()

        pipeline.add_component(name="query", instance=Passthrough(query=str, top_k=int))  # noqa
        pipeline.add_component(name="retriever", instance=InMemoryEmbeddingRetriever(document_store=self._document_store))  # noqa
        pipeline.add_component(name="result", instance=Passthrough(documents=list[Document]))  # noqa

        if self._with_hyde:
            self._use_hyde_embeddings(pipeline, "query.query", "retriever.query_embedding")
        else:
            self._use_query_embeddings(pipeline, "query.query", "retriever.query_embedding")

        if self._with_bm25:
            pipeline.add_component(name="joiner", instance=DocumentJoiner(join_mode="reciprocal_rank_fusion"))  # noqa

            self._add_bm25(pipeline, "query.query", "joiner.documents")

            pipeline.connect("query.top_k", "retriever.top_k")
            pipeline.connect("query.top_k", "joiner.top_k")
            pipeline.connect("retriever.documents", "joiner.documents")
            pipeline.connect("joiner.documents", "result.documents")
        else:
            pipeline.connect("query.top_k", "retriever.top_k")
            pipeline.connect("retriever.documents", "result.documents")

        if self._with_evaluator:
            self._add_evaluator(pipeline, "query.query", "result.documents")

        return pipeline


__all__ = (
    "get_or_create_document_store",
    "PipelineFactory"
)
