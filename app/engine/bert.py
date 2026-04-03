from pathlib import Path
from typing import override

import polars as pl

from app.engine.base import SearchEngine


class BERTSearchEngine(SearchEngine):
    def __init__(self, bert_model_path: Path):
        if not bert_model_path.exists():
            raise FileNotFoundError

        self.model = ...

    @override
    def load(self, documents: pl.DataFrame, index_col: str) -> None:
        ...

    @override
    def search(self, query: str, top_k: int) -> pl.DataFrame | None:
        ...
