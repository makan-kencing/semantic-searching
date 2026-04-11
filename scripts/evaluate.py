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


def f_score(b: int = 1, *, precision: float, recall: float) -> float:
    return (1 + b ** 2) * (precision * recall) / (b ** 2 * precision + recall)


@click.command()
@click.option("--dataset", "-i", type=click.Path(writable=True, dir_okay=False, path_type=Path), required=True)
@click.option("--output", "-o", type=click.Path(writable=True, dir_okay=False, path_type=Path), required=True)
@click.option("--iterations", "-n", default=20)
def evaluate(dataset: Path, output: Path, iterations: int):
    dataset: pl.DataFrame = pl.read_csv(dataset) \
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
        ))),
        "Development of mechanical machines and physical hardware": dataset.filter(pl.col("title").is_in((
            "Aluminium Can Crusher Machine",
            "Design and Build Delivery Robot",
            "Design of an Automated Powder Spray System for Cable Trunk",
            "Design and Development of Low-Cost Wind Powered Motorcycle Mobile Charger",
            "Design and Develop a Cost-Effective Hydroponics Autodoser System",
            "Arduino Based Dual-Axis Solar Tracking System",
            "Dielectric Strength Tester for Crude Palm Oil",
            "Disturbance Force Compensation Using Robust Controller in Linear Drive Positioning System",
            "Robot Control Interface with Video Feedback for Wireless Camera",
            "Investigate on the Breakdown Performance of Insulation Transformer Oil",
            "Investigating Energy Loss of Flow Over Non-Smooth Surfaces in an Open Channel Flow",
            "Investigating and Analyzing of Voltage Distributions Along High Voltage Insulators",
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
                try:
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
                    f.write(f"{average_semantic_similarity = }\n")
                except ZeroDivisionError:
                    f.write("Division by zero\n")
                finally:
                    f.write("\n")


if __name__ == '__main__':
    evaluate()
