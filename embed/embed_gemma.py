import argparse
import time
import os

# --- FIX 1: Prevent CPU "Traffic Jam" ---
# This must be set BEFORE importing numpy/torch.
# It forces each process to stick to 1 thread so they don't fight for resources.
# os.environ["OMP_NUM_THREADS"] = "1"

import datasets
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer

download_dir = ""
data_dir = ""


class Metadata:
    def __init__(self, short_name, split, text_col, label_col, multiple_per_entry):
        self.short_name = short_name
        self.split = split
        self.text_col = text_col
        self.label_col = label_col
        self.multiple_per_entry = multiple_per_entry


dataset_metadata = {
    "mteb/arxiv-clustering-p2p": Metadata("arxiv-clustering-p2p", "test", "sentences", "labels", True),
    "mteb/arxiv-clustering-s2s": Metadata("arxiv-clustering-s2s", "test", "sentences", "labels", True),
    "mteb/reddit-clustering": Metadata("reddit-clustering", "test", "sentences", "labels", True),
    "mteb/reddit-clustering-p2p": Metadata("reddit-clustering-p2p", "test", "sentences", "labels", True),
    "mteb/stackexchange-clustering": Metadata("stackexchange-clustering", "test", "sentences", "labels", True),
    "mteb/stackexchange-clustering-p2p": Metadata("stackexchange-clustering-p2p", "test", "sentences", "labels", True),
    "mteb/twentynewsgroups-clustering": Metadata("twentynewsgroups-clustering", "test", "sentences", "labels", True),
}


def embed(dataset_name):
    data = datasets.load_dataset(dataset_name, cache_dir=download_dir)

    # Use Google EmbedGemma
    model_name = "google/embeddinggemma-300m"

    # trust_remote_code=True is REQUIRED for EmbedGemma
    embedder = SentenceTransformer(model_name, cache_folder=download_dir, trust_remote_code=True)

    metadata = dataset_metadata[dataset_name]
    documents = []
    ground_truths = []

    # --- FIX 2: Manually add the prompt ---
    # EmbedGemma needs this instruction. We add it here because encode_multi_process
    # sometimes ignores the 'prompt_name' argument.
    instruction = "task: clustering | query: "

    for document in data[metadata.split]:
        gt = document[metadata.label_col]
        text = document[metadata.text_col]

        if not metadata.multiple_per_entry:
            gt = [gt]
            text = [text]

        ground_truths += gt
        # Prepend instruction to every document string
        documents += [instruction + t for t in text]

    ground_truths = pd.factorize(ground_truths)[0]
    print(len(documents), flush=True)
    print(len(ground_truths), flush=True)

    start = time.time()

    # Your original multi-process code
    # We explicitly tell it to use CPU (optional, but safer)
    pool = embedder.start_multi_process_pool(target_devices=['cpu'] * 5)  # Uses 5 cores. Adjust number as needed.

    # We pass the modified documents list
    embeddings_np = embedder.encode_multi_process(documents, pool)

    embedder.stop_multi_process_pool(pool)

    print(embeddings_np.shape, flush=True)
    print(len(ground_truths), flush=True)
    print(time.time() - start, flush=True)

    dataset_dir = f"{data_dir}/{metadata.short_name}"
    Path(dataset_dir).mkdir(parents=True, exist_ok=True)
    np.save(f"{dataset_dir}/{metadata.short_name}.npy", embeddings_np)
    with open(f"{dataset_dir}/{metadata.short_name}.gt", "w") as f:
        for g in ground_truths:
            f.write(f"{g} \n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name")
    parser.add_argument("--download_dir")
    parser.add_argument("--data_dir")
    args = parser.parse_args()

    download_dir = args.download_dir
    data_dir = args.data_dir

    embed(args.dataset_name)
