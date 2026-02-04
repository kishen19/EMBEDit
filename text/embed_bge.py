import argparse
import os
import time
from pathlib import Path

import datasets
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# --- DATASET REGISTRY ---
class Metadata:
    def __init__(
        self, short_name, split, text_col, label_col, multiple_per_entry, name=None
    ):
        self.short_name = short_name
        self.split = split
        self.text_col = text_col
        self.label_col = label_col
        self.multiple_per_entry = multiple_per_entry
        self.name = name


dataset_info = {
    "wikipedia_medium": (
        "BASF-AI/WikipediaMedium5Clustering",
        Metadata("wikipedia_medium", "test", "sentences", "labels", True),
    ),
    "wikipedia_easy": (
        "BASF-AI/WikipediaEasy10Clustering",
        Metadata("wikipedia_easy", "test", "sentences", "labels", True),
    ),
    "alloprof": (
        "mteb/AlloProfClusteringS2S.v2",
        Metadata("alloprof", "test", "sentences", "labels", False),
    ),
    "alloprof_p2p": (
        "mteb/AlloProfClusteringP2P.v2",
        Metadata("alloprof_p2p", "test", "sentences", "labels", False),
    ),
    "trec_covid": (
        "Uri-ka/ClusTREC-Covid",
        Metadata("trec_covid", "test", "sentences", "labels", False, name="title"),
    ),
    "trec_covid_p2p": (
        "Uri-ka/ClusTREC-Covid",
        Metadata(
            "trec_covid_p2p",
            "test",
            "sentences",
            "labels",
            False,
            name="title and abstract",
        ),
    ),
    "cities_wiki": (
        "jinaai/cities_wiki_clustering",
        Metadata("cities_wiki", "test", "sentences", "labels", True),
    ),
    "builtbench": (
        "mehrzad-shahin/BuiltBench-clustering-s2s",
        Metadata("builtbench", "test", "sentences", "labels", True),
    ),
    "builtbench_p2p": (
        "mehrzad-shahin/BuiltBench-clustering-p2p",
        Metadata("builtbench_p2p", "test", "sentences", "labels", True),
    ),
    "medrxiv": (
        "mteb/medrxiv-clustering-s2s",
        Metadata("medrxiv", "test", "sentences", "labels", False),
    ),
    "medrxiv_p2p": (
        "mteb/medrxiv-clustering-p2p",
        Metadata("medrxiv_p2p", "test", "sentences", "labels", False),
    ),
    "hal": (
        "lyon-nlp/clustering-hal-s2s",
        Metadata("hal", "test", "title", "domain", False, name="mteb_eval"),
    ),
    "biorxiv": (
        "mteb/biorxiv-clustering-s2s",
        Metadata("biorxiv", "test", "sentences", "labels", False),
    ),
    "biorxiv_p2p": (
        "mteb/biorxiv-clustering-p2p",
        Metadata("biorxiv_p2p", "test", "sentences", "labels", False),
    ),
    "twentynewsgroups": (
        "mteb/twentynewsgroups-clustering",
        Metadata("twentynewsgroups", "test", "sentences", "labels", True),
    ),
    "big_patent": (
        "mteb/big-patent",
        Metadata("big_patent", "test", "sentences", "labels", False),
    ),
    "stackexchange": (
        "mteb/stackexchange-clustering",
        Metadata("stackexchange", "test", "sentences", "labels", True),
    ),
    "stackexchange_p2p": (
        "mteb/stackexchange-clustering-p2p",
        Metadata("stackexchange_p2p", "test", "sentences", "labels", True),
    ),
    "reddit": (
        "mteb/reddit-clustering",
        Metadata("reddit", "test", "sentences", "labels", True),
    ),
    "reddit_p2p": (
        "mteb/reddit-clustering-p2p",
        Metadata("reddit_p2p", "test", "sentences", "labels", True),
    ),
    "arxiv": (
        "mteb/arxiv-clustering-s2s",
        Metadata("arxiv", "test", "sentences", "labels", True),
    ),
    "arxiv_p2p": (
        "mteb/arxiv-clustering-p2p",
        Metadata("arxiv_p2p", "test", "sentences", "labels", True),
    ),
}


def process_dataset(dataset_input, model_dir, output_dir):
    if dataset_input not in dataset_info:
        print(f"Error: {dataset_input} not in registry.")
        return

    dataset_key, metadata = dataset_info[dataset_input]

    # 1. Model Configuration
    model_name = "BAAI/bge-base-en-v1.5"
    has_gpu = torch.cuda.is_available()

    model = SentenceTransformer(model_name, cache_folder=model_dir)

    # BGE-base has a hard limit of 512 tokens.
    # Attempting to set this higher has no effect on this specific architecture.
    model.max_seq_length = 512
    print(
        f"Loaded {model_name}. Base dimension: {model.get_sentence_embedding_dimension()}"
    )

    # 2. Data Preparation
    ds = datasets.load_dataset(dataset_key, name=metadata.name, cache_dir=model_dir)
    split = metadata.split if metadata.split in ds else list(ds.keys())[0]
    data = ds[split]

    print(f"Loading {metadata.short_name}...")
    texts = [
        t
        for item in tqdm(data, desc="Texts")
        for t in (
            item[metadata.text_col]
            if metadata.multiple_per_entry
            else [item[metadata.text_col]]
        )
    ]
    labels = [
        l
        for item in tqdm(data, desc="Labels")
        for l in (
            item[metadata.label_col]
            if metadata.multiple_per_entry
            else [item[metadata.label_col]]
        )
    ]

    # 3. Multi-Process Execution
    # Since BGE-base is very small (438MB), you can run a high number of workers.
    # Spawning 32-48 workers is very efficient on your 96-core server.
    num_workers = torch.cuda.device_count() if has_gpu else 32
    pool = model.start_multi_process_pool(
        target_devices=None if has_gpu else ["cpu"] * num_workers
    )

    print(f"Embedding {len(texts)} documents using {num_workers} workers...")
    start_time = time.time()
    embeddings_full = model.encode_multi_process(
        texts, pool, batch_size=256 if has_gpu else 128
    )
    model.stop_multi_process_pool(pool)
    print(f"Completed in {time.time() - start_time:.2f}s")

    # 4. Save
    save_dir = Path(output_dir) / metadata.short_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # BGE-base-v1.5 outputs 768-dimensional vectors.
    # While it doesn't officially support Matryoshka like the v5 Stella/Qwen3,
    # it is often benchmarked at its full 768 dimensions.
    np.save(save_dir / f"{metadata.short_name}.npy", embeddings_full)

    gt_factors = pd.factorize(labels)[0]
    with open(save_dir / f"{metadata.short_name}.gt", "w") as f:
        f.write("\n".join(map(str, gt_factors)) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model_dir", default=os.path.expanduser("~/.cache"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    process_dataset(args.dataset, args.model_dir, args.output)
