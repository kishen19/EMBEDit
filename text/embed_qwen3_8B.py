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


# --- METADATA & FULL MAPPING LOGIC ---
class Metadata:
    def __init__(
        self,
        short_name,
        split,
        text_col,
        label_col,
        multiple_per_entry,
        name=None,
        language="eng",
    ):
        self.short_name = short_name
        self.split = split
        self.text_col = text_col
        self.label_col = label_col
        self.multiple_per_entry = multiple_per_entry
        self.name = name
        self.language = language


dataset_info = {
    # Done
    "wikipedia_medium": (
        "BASF-AI/WikipediaMedium5Clustering",
        Metadata("wikipedia_medium", "test", "sentences", "labels", True),
    ),
    # Done
    "wikipedia_easy": (
        "BASF-AI/WikipediaEasy10Clustering",
        Metadata("wikipedia_easy", "test", "sentences", "labels", True),
    ),
    # Done
    "alloprof": (
        "mteb/AlloProfClusteringS2S.v2",
        Metadata("alloprof", "test", "sentences", "labels", False, language="fra"),
    ),
    "alloprof_p2p": (
        "mteb/AlloProfClusteringP2P.v2",
        Metadata("alloprof_p2p", "test", "sentences", "labels", False, language="fra"),
    ),
    # Done
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
    # Done
    "builtbench": (
        "mehrzad-shahin/BuiltBench-clustering-s2s",
        Metadata("builtbench", "test", "sentences", "labels", True),
    ),
    "builtbench_p2p": (
        "mehrzad-shahin/BuiltBench-clustering-p2p",
        Metadata("builtbench_p2p", "test", "sentences", "labels", True),
    ),
    # Done
    "medrxiv": (
        "mteb/medrxiv-clustering-s2s",
        Metadata("medrxiv", "test", "sentences", "labels", False),
    ),
    "medrxiv_p2p": (
        "mteb/medrxiv-clustering-p2p",
        Metadata("medrxiv_p2p", "test", "sentences", "labels", False),
    ),
    # Done
    "hal": (
        "lyon-nlp/clustering-hal-s2s",
        Metadata(
            "hal", "test", "title", "domain", False, name="mteb_eval", language="fra"
        ),
    ),
    # Done
    "biorxiv": (
        "mteb/biorxiv-clustering-s2s",
        Metadata("biorxiv", "test", "sentences", "labels", False),
    ),
    "biorxiv_p2p": (
        "mteb/biorxiv-clustering-p2p",
        Metadata("biorxiv_p2p", "test", "sentences", "labels", False),
    ),
    # Done
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
        print(f"Dataset '{dataset_input}' not found.")
        return

    dataset_key, metadata = dataset_info[dataset_input]

    # 1. Initialize Model with CPU/GPU Auto-Detection
    has_gpu = torch.cuda.is_available()
    model_name = "Qwen/Qwen3-Embedding-8B"

    # Use float16 on GPU for speed; float32 on CPU for compatibility
    torch_dtype = "float16" if has_gpu else "float32"

    model = SentenceTransformer(
        model_name,
        cache_folder=model_dir,
        trust_remote_code=True,
        model_kwargs={"torch_dtype": torch_dtype},
    )

    # Set the sequence length for P2P tasks
    model.max_seq_length = 4096
    print(
        f"Model loaded. Max Seq Length: {model.max_seq_length} | Precision: {torch_dtype}"
    )

    # 2. Data Preparation
    ds = datasets.load_dataset(dataset_key, name=metadata.name, cache_dir=model_dir)
    # Support for common splits (test, validation, train fallback)
    split = metadata.split if metadata.split in ds else list(ds.keys())[0]
    data = ds[split]

    texts, labels = [], []
    for item in tqdm(data, desc="Loading Data"):
        t, l = item[metadata.text_col], item[metadata.label_col]
        if metadata.multiple_per_entry:
            texts.extend(t)
            labels.extend(l)
        else:
            texts.append(t)
            labels.append(l)

    # 3. Multi-Process Execution (CPU Fallback included)
    # If no target_devices passed, it uses all CUDA devices.
    # If no CUDA found, it spawns CPU processes.
    target_devices = (
        None if has_gpu else ["cpu"] * 8
    )  # Limit CPU processes for 8B model

    print(f"Starting multi-process pool on {'GPU' if has_gpu else 'CPU'}...")
    pool = model.start_multi_process_pool(target_devices=target_devices)

    start_time = time.time()
    embeddings_full = model.encode_multi_process(
        texts, pool, batch_size=128 if has_gpu else 8, chunk_size=500
    )
    model.stop_multi_process_pool(pool)
    print(f"Finished in {time.time() - start_time:.2f}s")

    # 4. Matryoshka & Save
    save_dir = Path(output_dir) / metadata.short_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1024-dim version
    embeddings_1024 = embeddings_full[:, :1024]
    norms = np.linalg.norm(embeddings_1024, axis=1, keepdims=True)
    embeddings_1024 /= np.where(norms == 0, 1e-10, norms)

    np.save(save_dir / f"{metadata.short_name}.npy", embeddings_full)
    np.save(save_dir / f"{metadata.short_name}_1024.npy", embeddings_1024)

    # .gt file format: one label per line
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
