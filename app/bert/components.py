import itertools
from dataclasses import replace
from typing import Iterable, Sequence

import numpy as np
import torch
from haystack import Document
from haystack.core.component import component
from tqdm import tqdm
from transformers import BertTokenizer, BertModel


class BERTEmbedder:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.model = BertModel.from_pretrained('bert-base-cased').to(self.device)

    def compile_embeddings(self, texts: Sequence[str], batch_size: int) -> list[np.ndarray]:
        embeddings = []

        # Process in batches to prevent memory errors
        with torch.no_grad():
            for batch_texts in tqdm(itertools.batched(texts, batch_size), total=len(texts)):
                # Tokenize
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(self.device)

                outputs = self.model(**inputs)

                # Extract the [CLS] token (index 0) for the whole batch
                # vector represents the summary of the Title + Abstract
                cls_vectors = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(cls_vectors)

        return list(itertools.chain.from_iterable(embeddings))


@component
class BERTTextEmbedder(BERTEmbedder):
    @component.output_types(embedding=list[float])
    def run(self, text: str) -> dict[str, list[float]]:
        if not isinstance(text, str):
            raise TypeError("BERTTextEmbedder expects a string as input.")

        embeddings = self.compile_embeddings((text,), 1)[0]

        return {"embedding": embeddings.tolist()}


@component
class BERTDocumentEmbedder(BERTEmbedder):
    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document], batch_size: int = 16) -> dict[str, list[Document]]:
        if not isinstance(documents, list) or documents and not isinstance(documents[0], Document):
            raise TypeError("BERTDocumentEmbedder expects a list of Documents as input.")

        embeddings = self.compile_embeddings([doc.content or "" for doc in documents], batch_size)

        new_documents = []
        for doc, emb in zip(documents, np.vstack(embeddings), strict=True):
            new_documents.append(replace(doc, embedding=emb.tolist()))

        return {"documents": new_documents}


__all__ = (
    "BERTTextEmbedder",
    "BERTDocumentEmbedder"
)