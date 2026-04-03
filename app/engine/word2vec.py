import logging
from pathlib import Path
from typing import override

import numpy as np
import polars as pl
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from app.engine.base import SearchEngine


logger = logging.getLogger("uvicorn.error")

class Word2VecSearchEngine(SearchEngine):
    def __init__(self, word2vec_model_path: Path):
        if not word2vec_model_path.exists():
            raise FileNotFoundError

        logger.info("Model: Loading Word2Vec model")
        self.model: KeyedVectors = KeyedVectors.load_word2vec_format(str(word2vec_model_path), binary=True)
        self.documents: pl.DataFrame | None = None
        self.document_vectors: np.ndarray | None = None

    @override
    def load(self, documents: pl.DataFrame, index_col: str) -> None:
        with logging_redirect_tqdm(loggers=[logger]):
            with tqdm(desc="Creating Word2Vec document embeddings", total=documents.height) as pbar:
                tfidf_vectorizer = TfidfVectorizer(
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=2
                )
                tfidf_matrix = tfidf_vectorizer.fit_transform(documents[index_col])
                word_to_idx = tfidf_vectorizer.vocabulary_

                def get_weighted_vector(text: str, row_idx: int):
                    vectors = []
                    weights = []

                    for word in text.lower().split():
                        if word in self.model and word in word_to_idx:
                            w_idx = word_to_idx[word]
                            weight = tfidf_matrix[row_idx, w_idx]

                            vectors.append(self.model[word] * weight)
                            weights.append(weight)

                    if not vectors or sum(weights) == 0:
                        return np.zeros(300)

                    pbar.update(1)
                    return np.sum(vectors, axis=0) / np.sum(weights)

                self.documents = documents
                self.document_vectors = np.array([
                    get_weighted_vector(text, i) for i, text in enumerate(documents[index_col])
                ])

    @override
    def search(self, query: str, top_k: int) -> pl.DataFrame | None:
        if self.documents is None or self.document_vectors is None:
            raise RuntimeError("Document not loaded.")

        query_words = query.lower().split()
        query_vectors = [self.model[word] for word in query_words if word in self.model]
        query_vectors = np.mean(query_vectors, axis=0).reshape(1, -1)

        scores = cosine_similarity(query_vectors, self.document_vectors)[0]

        return self.documents.with_columns(
            pl.Series("score", scores)
        ).sort(
            "score", descending=True
        ).head(top_k)
