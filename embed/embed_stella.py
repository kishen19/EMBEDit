import argparse
import time
import math
import os
from pathlib import Path

# Keeping your preference: OMP threads line is commented out.
# os.environ["OMP_NUM_THREADS"] = "1"

import datasets
import numpy as np
import pandas as pd
import multiprocessing
from sentence_transformers import SentenceTransformer

download_dir = ""
data_dir = ""


# --- WORKER FUNCTION (Must be at top level) ---
# This runs inside each separate process.
# It loads the model fresh to avoid "ModuleNotFoundError".
def embed_worker(documents_chunk, model_name, cache_folder):
    # Load the model inside the worker process
    # device='cpu' ensures it doesn't try to grab GPU memory if available
    worker_model = SentenceTransformer(
        model_name, cache_folder=cache_folder, trust_remote_code=True, device="cpu"
    )

    # Stella v5 needs no specific prompt for document clustering
    embeddings = worker_model.encode(
        documents_chunk,
        batch_size=32,  # Batch size within the worker
        show_progress_bar=False,
        convert_to_numpy=True,
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

    # Model configuration
    model_name = "dunzhang/stella_en_1.5B_v5"

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
    print(f"Labels: {len(ground_truths)}", flush=True)

    start = time.time()

    # --- MANUAL MULTIPROCESSING SETUP ---
    # We split documents into chunks and process them in parallel.
    # Adjust processes based on your RAM. Stella is 1.5B (~3GB RAM per process).
    num_processes = 5
    chunk_size = math.ceil(len(documents) / num_processes)

    # Create chunks of documents
    chunks = [
        documents[i : i + chunk_size] for i in range(0, len(documents), chunk_size)
    ]

    print(
        f"Spawning {num_processes} workers for {len(documents)} documents...",
        flush=True,
    )

    # Use 'spawn' context to ensure clean process start (fixes some threading issues)
    ctx = multiprocessing.get_context("spawn")

    with ctx.Pool(processes=num_processes) as pool:
        # We pass the model NAME, not the object.
        results = pool.starmap(
            embed_worker, [(chunk, model_name, download_dir) for chunk in chunks]
        )

    # Concatenate results from all workers
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

    # Explicitly set start method to spawn to avoid forking issues with HuggingFace
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass

    embed(args.dataset_name)
