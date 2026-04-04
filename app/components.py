from typing import Any

import numpy as np
from haystack import component, Document
from txtai.pipeline import Similarity


@component
class SimilarityEvaluator:
    def __init__(self, model: str):
        self.similarity = Similarity(model)

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


__all__ = (
    "SimilarityEvaluator",
    "Passthrough",
    "HypotheticalDocumentEmbedder"
)
