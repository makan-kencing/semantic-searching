from dataclasses import replace
from typing import Any

import numpy as np
from haystack import component, Document
from sentence_transformers import SentenceTransformer
from txtai.pipeline import Similarity


@component
class SimilarityEvaluator:
    def __init__(self, model: Similarity):
        self.similarity = model

    @component.output_types(scores=list[float])
    def run(self, query: str, documents: list[Document]) -> dict[str, list[float]]:
        return {"scores": list(map(
            lambda t: t[1],
            sorted(self.similarity(query, [doc.content for doc in documents]), key=lambda t: t[0])
        ))}


@component
class Passthrough:
    def __init__(self, **attrs: type):
        component.set_input_types(self, **attrs)
        component.set_output_types(self, **attrs)

    @staticmethod
    def run(**kwargs) -> dict[str, Any]:
        return kwargs


@component
class HypotheticalDocumentEmbedder:
    @component.output_types(hypothetical_embedding=list[float])
    def run(self, documents: list[Document]):
        stacked_embeddings = np.array([doc.embedding for doc in documents])
        avg_embeddings = np.mean(stacked_embeddings, axis=0)
        hyde_vector = avg_embeddings.reshape((1, len(avg_embeddings)))
        return {"hypothetical_embedding": hyde_vector[0].tolist()}


class SentenceTransformerEmbedder:
    def __init__(self, model: SentenceTransformer):
        self.model = model


@component
class SentenceTransformerQueryEmbedder(SentenceTransformerEmbedder):
    @component.output_types(embedding=list[float])
    def run(self, text: str) -> dict[str, list[float]]:
        if not isinstance(text, str):
            raise TypeError("SentenceTransformerQueryEmbedder expects a string as input.")

        embeddings = self.model.encode_query(text, show_progress_bar=True)

        return {"embedding": embeddings.tolist()}


@component
class SentenceTransformerDocumentEmbedder(SentenceTransformerEmbedder):
    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]) -> dict[str, list[Document]]:
        if not isinstance(documents, list) or documents and not isinstance(documents[0], Document):
            raise TypeError("SentenceTransformerDocumentEmbedder expects a list of Documents as input.")

        embeddings = self.model.encode_document([doc.content or "" for doc in documents], show_progress_bar=True)

        new_documents = []
        for doc, emb in zip(documents, embeddings, strict=True):
            new_documents.append(replace(doc, embedding=emb.tolist()))

        return {"documents": new_documents}


__all__ = (
    "SimilarityEvaluator",
    "Passthrough",
    "HypotheticalDocumentEmbedder",
    "SentenceTransformerQueryEmbedder",
    "SentenceTransformerDocumentEmbedder"
)
