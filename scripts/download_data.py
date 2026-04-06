#!/usr/bin/env python
import asyncio
import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterable

import aiohttp
import click
from bs4 import BeautifulSoup
from tqdm import tqdm
from yarl import URL

logging.basicConfig()
logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class Document:
    title: str
    abstract: str | None
    webpage_url: URL
    document_url: URL | None


class EPrintsScraper:
    def __init__(self, session: aiohttp.ClientSession, *, host: URL):
        self.session = session
        self.host = host

    async def fetch_all(self) -> AsyncIterable[Document]:
        search_offset = 0
        with tqdm(desc="Downloading FYP data.") as pbar:
            while True:
                params = {
                    "type": "other",
                    "search_offset": search_offset
                }
                async with self.session.get(self.host / "cgi" / "search" / "archive" / "advanced",
                                            params=params) as ret:
                    ret.raise_for_status()

                    soup = BeautifulSoup(await ret.read(), "lxml")

                    span_current, span_last, span_total = soup.select(".ep_search_controls .ep_search_number")
                    if not pbar.total:
                        pbar.total = int(span_total.text)

                    anchors = soup.select(".ep_search_results .ep_paginate_list .ep_search_result td:nth-of-type(2) a")

                    if not anchors:
                        break

                    for anchor in anchors:
                        doc_id = anchor.get("href").split("/")[-2]
                        yield await self.fetch(doc_id)
                        pbar.update(1)

                    search_offset += 20
                    if int(span_last.text) > search_offset:
                        break

    async def fetch(self, doc_id: int | str) -> Document:
        async with self.session.get(self.host / f"{doc_id}" / "") as ret:
            ret.raise_for_status()

            soup = BeautifulSoup(await ret.read(), "lxml")

            abstract_p = soup.select_one(".ep_summary_content_main p:nth-of-type(2)")
            document_a = soup.select_one(".ep_document_link")
            return Document(
                title=soup.select_one(".ep_tm_pagetitle").get_text(strip=True).replace("\r\n", " "),
                abstract=abstract_p.text.replace("\r\n", " ") if abstract_p else None,
                webpage_url=self.host / f"{doc_id}" / "",
                document_url=URL(document_a.get("href")) if document_a else None  # noqa
            )

    async def download(self, document: Document) -> tuple[str, bytes]:
        if document.document_url is None:
            raise ValueError()

        async with self.session.get(document.document_url) as ret:
            ret.raise_for_status()
            if ret.content_disposition is None:
                raise RuntimeError()
            return ret.content_disposition.filename or f"{document.title}.pdf", await ret.read()


async def download_fyp_dataset(out_dir: Path) -> None:
    output_file = out_dir / "fyp.csv"
    if output_file.exists():
        return

    with output_file.open("w") as f:
        writer = csv.writer(f)
        writer.writerow(("title", "abstract", "webpage_url", "document_url"))

        async with aiohttp.ClientSession() as session:
            scraper = EPrintsScraper(session, host=URL("https://eprints.tarc.edu.my/"))

            async for doc in scraper.fetch_all():
                writer.writerow((doc.title, doc.abstract, doc.webpage_url, doc.document_url))


@click.command()
@click.option("--dataset-dir", type=click.Path(writable=True, file_okay=False, path_type=Path), required=True)
@click.option("--model-dir", type=click.Path(writable=True, file_okay=False, path_type=Path), required=True)
def download(dataset_dir: Path, model_dir: Path):
    dataset_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    asyncio.run(download_fyp_dataset(dataset_dir))


if __name__ == '__main__':
    download()
