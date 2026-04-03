from abc import abstractmethod
from typing import Protocol, Sequence
import polars as pl


class SearchEngine(Protocol):
    @abstractmethod
    def load(self, documents: pl.DataFrame, index_col: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(self, query: str, top_k: int) -> pl.DataFrame | None:
        raise NotImplementedError
