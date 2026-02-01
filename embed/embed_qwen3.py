import argparse
import time
import math
import os
import multiprocessing
import torch
import numpy as np
import pandas as pd
import datasets
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# --- HARDWARE OPTIMIZATION ---
# With 96 physical cores and 16 processes, each process gets 6 physical cores.
# We set OMP threads to 6 to fully utilize the CPU without oversubscription.
os.environ["OMP_NUM_THREADS"] = "48"
os.environ["MKL_NUM_THREADS"] = "48"
os.environ["OPENBLAS_NUM_THREADS"] = "48"

download_dir = ""
data_dir = ""


# --- WORKER FUNCTION ---
def embed_worker(documents_chunk, model_name, cache_folder):
    # Load model on CPU
    # torch_dtype="auto" will use bfloat16/float16 if supported, speeding up AVX512 ops
    worker_model = SentenceTransformer(
        model_name,
        cache_folder=cache_folder,
        trust_remote_code=True,
        device="cpu",
        model_kwargs={"torch_dtype": "auto"},
    )

    # Qwen3-8B: No instruction needed for document clustering.
    # Batch size can be larger (32-64) because you have massive RAM.
    embeddings = worker_model.encode(
        documents_chunk, batch_size=32, show_progress_bar=True, convert_to_numpy=True
    )
    return embeddings


class Metadata:
    def __init__(self, short_name, split, text_col, label_col, multiple_per_entry):
        self.short_name = short_name
        self.split = split
        self.text_col = text_col
        self.label_col = label_col
        self.multiple_per_entry = multiple_per_entry


dataset_metadata = {
    "mteb/arxiv-clustering-p2p": Metadata(
        "arxiv-clustering-p2p", "test", "sentences", "labels", True
    ),
    "mteb/arxiv-clustering-s2s": Metadata(
        "arxiv-clustering-s2s", "test", "sentences", "labels", True
    ),
    "mteb/reddit-clustering": Metadata(
        "reddit-clustering", "test", "sentences", "labels", True
    ),
    "mteb/reddit-clustering-p2p": Metadata(
        "reddit-clustering-p2p", "test", "sentences", "labels", True
    ),
    "mteb/stackexchange-clustering": Metadata(
        "stackexchange-clustering", "test", "sentences", "labels", True
    ),
    "mteb/stackexchange-clustering-p2p": Metadata(
        "stackexchange-clustering-p2p", "test", "sentences", "labels", True
    ),
    "mteb/twentynewsgroups-clustering": Metadata(
        "twentynewsgroups-clustering", "test", "sentences", "labels", True
    ),
}


def embed(dataset_name):
    print(f"Loading dataset: {dataset_name}...", flush=True)
    data = datasets.load_dataset(dataset_name, cache_dir=download_dir)

    # --- MODEL: Qwen3-Embedding-8B ---
    # The current SOTA for open-weights embedding models.
    # Output Dimension: 4096 (Note: Files will be 4x larger than standard models)
    model_name = "Qwen/Qwen3-Embedding-0.6B"

    metadata = dataset_metadata[dataset_name]
    documents = []
    ground_truths = []

    for document in data[metadata.split]:
        gt = document[metadata.label_col]
        text = document[metadata.text_col]

        if not metadata.multiple_per_entry:
            gt = [gt]
            text = [text]

        ground_truths += gt
        documents += text

    ground_truths = pd.factorize(ground_truths)[0]
    print(f"Documents: {len(documents)}", flush=True)

    start = time.time()

    # --- PROCESS CONFIGURATION ---
    # With 1.5TB RAM, we are not limited by memory.
    # We choose 16 processes to balance startup time vs throughput.
    # 16 processes * 6 threads = 96 cores (Perfect utilization of physical cores).
    num_processes = 2

    chunk_size = math.ceil(len(documents) / num_processes)
    chunks = [
        documents[i : i + chunk_size] for i in range(0, len(documents), chunk_size)
    ]

    print(
        f"Spawning {num_processes} workers (utilizing 96 physical cores)...", flush=True
    )

    ctx = multiprocessing.get_context("spawn")

    with ctx.Pool(processes=num_processes) as pool:
        results = pool.starmap(
            embed_worker,
            tqdm(
                [(chunk, model_name, download_dir) for chunk in chunks],
                desc="Embedding",
                total=len(chunks),
            ),
        )

    embeddings_np = np.concatenate(results, axis=0)

    print(f"Embedding Shape: {embeddings_np.shape}", flush=True)
    print(f"Time taken: {time.time() - start:.2f}s", flush=True)

    dataset_dir = f"{data_dir}/{metadata.short_name}"
    Path(dataset_dir).mkdir(parents=True, exist_ok=True)
    np.save(f"{dataset_dir}/{metadata.short_name}.npy", embeddings_np)
    with open(f"{dataset_dir}/{metadata.short_name}.gt", "w") as f:
        for g in ground_truths:
            f.write(f"{g} \n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--download_dir", default="./cache")
    parser.add_argument("--data_dir", default="./data")
    args = parser.parse_args()

    download_dir = args.download_dir
    data_dir = args.data_dir

    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass

    embed(args.dataset_name)
