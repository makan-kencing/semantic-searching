from dataclasses import replace

from haystack import component, Document
from sentence_transformers import SentenceTransformer


class LLMEmbedder:
    def __init__(self, model: str = "Qwen/Qwen3-Embedding-0.6B", device: str | None = None):
        self.model = SentenceTransformer(model, device=device)


@component
class LLMQueryEmbedder(LLMEmbedder):
    @component.output_types(embedding=list[float])
    def run(self, text: str) -> dict[str, list[float]]:
        if not isinstance(text, str):
            raise TypeError("LLMTextEmbedder expects a string as input.")

        embeddings = self.model.encode_query(text, show_progress_bar=True)

        return {"embedding": embeddings.tolist()}


@component
class LLMDocumentEmbedder(LLMEmbedder):
    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]) -> dict[str, list[Document]]:
        if not isinstance(documents, list) or documents and not isinstance(documents[0], Document):
            raise TypeError("LLMDocumentEmbedder expects a list of Documents as input.")

        embeddings = self.model.encode_document([doc.content or "" for doc in documents], show_progress_bar=True)

        new_documents = []
        for doc, emb in zip(documents, embeddings, strict=True):
            new_documents.append(replace(doc, embedding=emb.tolist()))

        return {"documents": new_documents}


__all__ = (
    "LLMQueryEmbedder",
    "LLMDocumentEmbedder"
)
