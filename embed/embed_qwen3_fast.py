import argparse
import time
import os
import math
import multiprocessing
from tqdm import tqdm

# --- CRITICAL CPU SETTINGS ---
# For multi-process encoding, preventing thread contention is vital for speed.
# We limit each process to a single thread to avoid oversubscription.
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"

import datasets
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer

download_dir = ""
data_dir = ""
worker_model = None


# --- WORKER INITIALIZATION ---
# This function is called once per worker process.
# It loads the model into a global variable for that process.
def init_worker(model_name, cache_folder):
    global worker_model
    worker_model = SentenceTransformer(
        model_name,
        cache_folder=cache_folder,
        trust_remote_code=True,
        model_kwargs={"torch_dtype": "auto"},
    )


# --- WORKER FUNCTION ---
# This function runs in the worker process and uses the pre-loaded model.
def embed_worker(documents_chunk):
    global worker_model
    return worker_model.encode(documents_chunk, batch_size=32, convert_to_numpy=True)


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


def embed(dataset_name, num_processes):
    print(f"Loading dataset: {dataset_name}...", flush=True)
    data = datasets.load_dataset(dataset_name, cache_dir=download_dir)

    # --- MODEL: Qwen/Qwen3-Embedding-0.6B ---
    model_name = "Qwen/Qwen3-Embedding-0.6B"

    metadata = dataset_metadata[dataset_name]
    documents = []
    ground_truths = []

    # Qwen3: No instruction needed for document clustering (symmetric task).
    for document in data[metadata.split]:
        gt = document[metadata.label_col]
        text = document[metadata.text_col]

        if not metadata.multiple_per_entry:
            gt = [gt]
            text = [text]

        ground_truths += gt
        documents += text

    ground_truths = pd.factorize(ground_truths)[0]
    print(f"Documents to embed: {len(documents)}", flush=True)

    start = time.time()

    # --- EFFICIENT MULTIPROCESSING WITH PROGRESS BAR ---
    # We manually create a multiprocessing pool to get progress updates.
    # The initializer loads the model once per worker, avoiding slowdowns.
    chunk_size = math.ceil(
        len(documents) / (num_processes * 4)
    )  # Make chunks smaller for better progress bar updates
    chunks = [
        documents[i : i + chunk_size] for i in range(0, len(documents), chunk_size)
    ]

    print(
        f"Spawning {num_processes} workers to process {len(chunks)} chunks...",
        flush=True,
    )

    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(
        processes=num_processes,
        initializer=init_worker,
        initargs=(model_name, download_dir),
    ) as pool:
        results = list(
            tqdm(
                pool.imap(embed_worker, chunks),
                total=len(chunks),
                desc="Embedding",
            )
        )

    embeddings_np = np.concatenate(results, axis=0)

    print(f"Embedding Shape: {embeddings_np.shape}", flush=True)
    print(f"Ground Truths: {len(ground_truths)}", flush=True)
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
    parser.add_argument(
        "--num_processes",
        type=int,
        default=4,
        help="Number of parallel processes to use for encoding.",
    )
    args = parser.parse_args()

    download_dir = args.download_dir
    data_dir = args.data_dir

    embed(args.dataset_name, args.num_processes)
