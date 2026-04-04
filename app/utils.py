from pathlib import Path

from haystack import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore


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


__all__ = (
    "get_or_create_document_store",
)
