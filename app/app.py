import logging
from contextlib import asynccontextmanager
from pathlib import Path

import polars as pl
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from huggingface_hub.errors import BadRequestError
from txtai.pipeline import Similarity

from app.engine.base import SearchEngine
from app.engine.word2vec import Word2VecSearchEngine
from app.engine.bert import BERTSearchEngine
from app.schemas import ApplicationContext

search_engines: dict[type[SearchEngine], SearchEngine] = {}
similarity: Similarity
logger = logging.getLogger("uvicorn.error")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global similarity

    logger.info("Dataset: Reading FYP dataset.")
    documents = pl.read_csv("./data/fyp.csv", has_header=True).filter(pl.col("abstract").is_not_null())

    logger.info("Engine: Loading Word2Vec engine.")
    word2vec = Word2VecSearchEngine(Path("./models/GoogleNews-vectors-negative300.bin"))
    word2vec.load(documents, "abstract")
    search_engines[Word2VecSearchEngine] = word2vec

    logger.info("Engine: Loading BERT engine.")
    bert = BERTSearchEngine(Path("./models/GoogleNews-vectors-negative300.bin"))
    bert.load(documents, "abstract")
    search_engines[BERTSearchEngine] = bert

    logger.info("Model: Loading similarity model.")
    similarity = Similarity("valhalla/distilbart-mnli-12-3")

    yield

app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html.j2", context={
        "context": ApplicationContext(available_engines=tuple(search_engines.keys()))
    })


@app.get("/hx-search", response_class=HTMLResponse)
async def search(request: Request, query: str, type: str):
    for clazz, engine in search_engines.items():
        if clazz.__name__ != type:
            continue

        documents = engine.search(query, 10)
        if len(documents) > 0:
            scores = similarity(query, documents["abstract"].to_list())
            scores = pl.DataFrame(scores, schema=["index", "similarity_metric"], orient="row").sort("index")
            documents = pl.concat((documents, scores), how="horizontal")
        return templates.TemplateResponse(request=request, name="search.html.j2", context={
            "documents": documents
        })
    raise BadRequestError()
