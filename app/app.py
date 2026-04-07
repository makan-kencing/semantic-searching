import itertools
import logging
import time
from contextlib import asynccontextmanager

import polars as pl
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from haystack import Pipeline, Document

from app.llm.pipeline import create as create_llm_pipeline
from app.sbert.pipeline import create as create_sbert_pipeline
from app.schemas import ApplicationContext
from app.word2vec.pipeline import create as create_word2vec_pipeline

logger = logging.getLogger("uvicorn.error")

logger.info("Dataset: Reading FYP dataset.")

dataset = pl.read_csv("./data/fyp.csv", has_header=True) \
    .filter(pl.col("abstract").is_not_null()) \
    .unique("title")
documents = [Document(id=row["title"], content=row["abstract"]) for row in dataset.iter_rows(named=True)]
pipelines: dict[str, Pipeline] = {}


def refresh_pipeline(use_hyde: bool = False, use_bm25: bool = False, use_evaluator: bool = False):
    for name, factory in (
            ("Word2Vec", create_word2vec_pipeline(documents)),
            ("Sentence BERT", create_sbert_pipeline(documents)),
            ("LLM", create_llm_pipeline(documents))
    ):
        if use_hyde:
            factory.with_hyde()
        if use_bm25:
            factory.with_bm25()
        if use_evaluator:
            factory.with_evaluator()

        logger.info(f"Loading {name} pipeline.")
        pipelines[name] = factory.make()


@asynccontextmanager
async def lifespan(_):
    refresh_pipeline()
    yield


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html.j2", context={
        "context": ApplicationContext(available_pipelines=tuple(pipelines.keys()))
    })


@app.get("/hx-search", response_class=HTMLResponse)
async def search(request: Request, query: str, type: str, top_k: int = 10):
    pipeline = pipelines.get(type)
    if pipeline is None:
        raise HTTPException(status_code=404)

    start_time = time.perf_counter()
    result = pipeline.run(data={
        "query": {"query": query, "top_k": top_k}
    }, include_outputs_from={"result", "evaluator"})
    end_time = time.perf_counter()

    return templates.TemplateResponse(request=request, name="search.html.j2", context={
        "time_s": end_time - start_time,
        "documents": list(itertools.zip_longest(
            result["result"]["documents"],
            result["evaluator"]["scores"] if "evaluator" in result else []
        ))
    })


@app.post("/hx-refresh")
def refresh(use_hyde: bool = False, use_bm25: bool = False, use_evaluator: bool = False) -> None:
    refresh_pipeline(use_hyde, use_bm25, use_evaluator)
