from pathlib import Path
from timeit import timeit

import click
import polars as pl
from haystack import Document
from haystack.components.evaluators import DocumentNDCGEvaluator, DocumentMAPEvaluator

from app.components import SimilarityEvaluator
from app.llm.pipeline import create as create_llm
from app.sbert.pipeline import create as create_sbert
from app.utils import PipelineFactory, similarity
from app.word2vec.pipeline import create as create_w2v

ndcg_eval = DocumentNDCGEvaluator()
map_eval = DocumentMAPEvaluator()
semantic_eval = SimilarityEvaluator(similarity)

iterations = 20


def f_score(b: int = 1, *, precision: float, recall: float) -> float:
    return (1 + b ** 2) * (precision * recall) / (b ** 2 * precision + recall)


@click.command()
@click.option("--dataset-path", type=click.Path(writable=True, dir_okay=False, path_type=Path), required=True)
@click.option("--output", "-o", type=click.Path(writable=True, dir_okay=False, path_type=Path), required=True)
def evaluate(dataset_path: Path, output: Path):
    dataset = pl.read_csv(dataset_path) \
        .filter(pl.col("abstract").is_not_null()) \
        .unique("title")
    grounded_truth_datasets: dict[str, pl.DataFrame] = {
        "Design of automated and robotic systems": dataset.filter(pl.col("title").is_in((
            "Robot Control Interface with Video Feedback for Wireless Camera",
            "Design and Build Delivery Robot",
            "Arduino Based Dual-Axis Solar Tracking System",
            "Identification in the Presence of Speed Bumps through Few-Shot Learning",
            "Robot Control Interface with Video Feedback for Wireless Camera",
            "Single-Sample Face Recognition for Attendance Record",
            "Cable Fault Recognition Using Image Analysis Of Phase Resolved Partial Discharge Pattern",
            "Banana Leaf Disease Detection Using Image Processing Methods",
            "Design of an Automated Powder Spray System for Cable Trunk",
            "Design and Develop a Cost-Effective Hydroponics Autodoser System",
        ))),
        "Biological diversity and biochemical analysis of natural substances": dataset.filter(pl.col("title").is_in((
            "Extraction Yield and Antimicrobial of Banana (Musa) Peel, Calamansi Lime (Citrus X Microcarpa) Peel and Durian (Durio Zibethinus) Husk Extracted by Using Different Solvents",
            "Evaluation of Antihyperglycaemic and Antimicrobial Effects of Different Extracts (Methanol, Ethanol, Chloroform and Aqueous) of Andrographis paniculata and Andrographis paniculata Herbal Tea",
            "Interaction of Apigenin and Hesperetin on Xanthine Oxidase Inhibition and Their Inhibitory Mechanism",
            "A Review on the Effect of Different Yeasts on the Fermentation of Red Dragon Fruit",
            "Bioactive Compounds and Physical Analysis of Bentong Gingers and Commercial Gingers Powder by Different Storage Temperature",
            "Effect of Lactic Acid Bacteria on the Physicochemical, Anti-Inflammatory Antioxidant and Microbiological Properties of Kombucha",
        )))
    }

    documents = [Document(id=row["title"], content=row["abstract"]) for row in dataset.iter_rows(named=True)]
    grounded_truth_documents: dict[str, list[Document]] = {
        query: [Document(id=row["title"], content=row["abstract"]) for row in dataset.iter_rows(named=True)]
        for query, dataset in grounded_truth_datasets.items()
    }

    algorithms: dict[str, PipelineFactory] = {
        "Word2Vec": create_w2v(documents),
        "BERT": create_sbert(documents),
        "LLM": create_llm(documents)
    }

    with (output.open("w") as f):
        for name, factory in algorithms.items():
            f.write("=" * 40 + "\n")
            f.write(f"Evaluating {name}.\n")
            f.write("=" * 40 + "\n")

            pipeline = factory.make()
            top_k = 10
            for query, ground_truth_documents in grounded_truth_documents.items():
                get_results = lambda: pipeline.run(data={
                    "query": {"query": query, "top_k": top_k}
                })

                runtime = timeit(get_results, number=iterations) / iterations * 1000
                results = get_results()

                retrieved_documents = results["result"]["documents"]
                relevant_documents = [doc for doc in results["result"]["documents"]
                                      if any(gt.id == doc.id for gt in ground_truth_documents)]

                f.write(f"{query = }\n")
                f.write("-" * 40 + "\n")

                f.write(f"{len(retrieved_documents) = }\n")
                f.write(f"{len(relevant_documents) = }\n")
                f.write(f"{runtime = }ms\n")

                precision = len(relevant_documents) / len(retrieved_documents)
                f.write(f"{precision = }\n")

                recall = len(relevant_documents) / len(ground_truth_documents)
                f.write(f"{recall = }\n")

                fallout = (len(retrieved_documents) - len(relevant_documents)) / \
                    (len(documents) - len(ground_truth_documents))
                f.write(f"{fallout = }\n")

                f_score_1 = f_score(1, precision=precision, recall=recall)
                f.write(f"{f_score_1 = }\n")

                f_score_2 = f_score(2, precision=precision, recall=recall)
                f.write(f"{f_score_2 = }\n")

                ndcg = ndcg_eval.run(
                    ground_truth_documents=[ground_truth_documents],
                    retrieved_documents=[results["result"]["documents"]]
                )["score"]
                f.write(f"{ndcg = }\n")

                ap = map_eval.run(
                    ground_truth_documents=[ground_truth_documents],
                    retrieved_documents=[results["result"]["documents"]]
                )["score"]
                f.write(f"{ap = }\n")

                semantic_similarities = semantic_eval.run(
                    query=query,
                    documents=retrieved_documents
                )["scores"]
                average_semantic_similarity = sum(semantic_similarities) / len(semantic_similarities)
                f.write(f"{similarity = }\n")

                f.write("\n")
        f.write("=" * 40 + "\n")


if __name__ == '__main__':
    evaluate()
