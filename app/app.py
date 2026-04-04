import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

import polars as pl
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from haystack import Pipeline, Document

from app.bert.pipeline import create as create_bert_pipeline
from app.sbert.pipeline import create as create_hybrid_pipeline
from app.schemas import ApplicationContext
from app.word2vec.pipeline import create as create_word2vec_pipeline

pipelines: dict[str, Pipeline] = {}
logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Dataset: Reading FYP dataset.")
    documents = pl.read_csv("./data/fyp.csv", has_header=True) \
        .filter(pl.col("abstract").is_not_null()) \
        .unique("title")
    documents = [Document(id=row["title"], content=row["abstract"]) for row in documents.iter_rows(named=True)]

    logger.info("Loading Word2Vec pipeline.")
    pipelines["Word2Vec"] = create_word2vec_pipeline(Path("./models/GoogleNews-vectors-negative300.bin"), documents)

    # logger.info("Loading BERT pipeline.")
    # pipelines["BERT"] = create_bert_pipeline(documents)

    logger.info("Loading SBERT pipeline.")
    pipelines["HyDE + SBERT + BM25"] = create_hybrid_pipeline(documents)

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
    })
    end_time = time.perf_counter()

    return templates.TemplateResponse(request=request, name="search.html.j2", context={
        "time_s": end_time - start_time,
        "documents": list(zip(result["result"]["documents"], result["result"]["similarity"]))
    })
