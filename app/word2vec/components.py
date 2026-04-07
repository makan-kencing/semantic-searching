import itertools
from dataclasses import replace
from typing import Iterable

import numpy as np
import gensim.downloader as api
from gensim.models import KeyedVectors
from haystack import component, Document
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm


class Word2VecEmbedder:
    def __init__(self, model: str):
        self.model: KeyedVectors = api.load(model)

    def get_embeddings(self, texts: Iterable[str]) -> list[np.ndarray]:
        t1, t2 = itertools.tee(texts, 2)
        tfidf_vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1
        )
        tfidf_matrix = tfidf_vectorizer.fit_transform(t1)
        word_to_idx = tfidf_vectorizer.vocabulary_

        def get_weighted_vector(text, row_idx):
            words = str(text).lower().split()
            vectors = []
            weights = []

            for word in words:
                if word in self.model and word in word_to_idx:
                    w_idx = word_to_idx[word]
                    weight = tfidf_matrix[row_idx, w_idx]

                    vectors.append(self.model[word] * weight)
                    weights.append(weight)

            if not vectors or sum(weights) == 0:
                return np.zeros(300)

            return np.sum(vectors, axis=0) / np.sum(weights)

        return list(tqdm(get_weighted_vector(text, i) for i, text in enumerate(t2)))


@component
class Word2VecTextEmbedder(Word2VecEmbedder):
    @component.output_types(embedding=list[float])
    def run(self, text: str) -> dict[str, list[float]]:
        if not isinstance(text, str):
            raise TypeError("Word2VecTextEmbedder expects a string as input.")

        embeddings = self.get_embeddings((text,))[0]

        return {"embedding": embeddings.tolist()}


@component
class Word2VecDocumentEmbedder(Word2VecEmbedder):
    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]) -> dict[str, list[Document]]:
        if not isinstance(documents, list) or documents and not isinstance(documents[0], Document):
            raise TypeError("Word2VecDocumentEmbedder expects a list of Documents as input.")

        embeddings = self.get_embeddings((doc.content for doc in documents))  # noqa

        new_documents = []
        for doc, emb in zip(documents, embeddings, strict=True):
            new_documents.append(replace(doc, embedding=emb.tolist()))

        return {"documents": new_documents}


__all__ = (
    "Word2VecTextEmbedder",
    "Word2VecDocumentEmbedder",
)
