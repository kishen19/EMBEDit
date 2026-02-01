import argparse
import time
import math
import os
import multiprocessing
import numpy as np
import pandas as pd
import datasets
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm  # <--- NEW IMPORT


# --- METADATA & MAPPING LOGIC ---
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


# Your provided dataset list
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
        Metadata(
            "alloprof",
            "test",
            "sentences",
            "labels",
            False,
            language="fra",
        ),
    ),
    "alloprof_p2p": (
        "mteb/AlloProfClusteringP2P.v2",
        Metadata(
            "alloprof_p2p",
            "test",
            "sentences",
            "labels",
            False,
            language="fra",
        ),
    ),
    "trec_covid": (
        "Uri-ka/ClusTREC-Covid",
        Metadata("trec_covid", "test", "sentences", "labels", False, name="title"),
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
        Metadata(
            "hal",
            "test",
            "title",
            "domain",
            False,
            name="mteb_eval",
            language="fra",
        ),
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


# --- WORKER FUNCTION ---
def embed_worker(args_tuple):
    """
    Runs inside a separate process. Loads the model and encodes a chunk of text.
    """
    # Unpack arguments manually
    documents_chunk, model_name, cache_folder = args_tuple

    # Load model on CPU
    worker_model = SentenceTransformer(
        model_name,
        cache_folder=cache_folder,
        trust_remote_code=True,
        device="cpu",
        model_kwargs={"torch_dtype": "auto"},
    )

    # Encode (Always generate full 4096 dims)
    embeddings = worker_model.encode(
        documents_chunk, batch_size=64, show_progress_bar=False, convert_to_numpy=True
    )

    return embeddings


def chunk_generator(
    dataset_iterable, metadata, chunk_size, model_args, label_accumulator
):
    chunk_doc = []

    for item in dataset_iterable:
        text = item[metadata.text_col]
        label = item[metadata.label_col]

        if not metadata.multiple_per_entry:
            text = [text]
            label = [label]

        chunk_doc.extend(text)
        label_accumulator.extend(label)

        if len(chunk_doc) >= chunk_size:
            yield (chunk_doc, *model_args)
            chunk_doc = []

    if chunk_doc:
        yield (chunk_doc, *model_args)


def embed_qwen3_8B(
    data_iterable,
    metadata,
    cache_folder,
    ground_truths_list,
    num_processes,
    total_samples=None,  # <--- NEW ARGUMENT
):
    model_name = "Qwen/Qwen3-Embedding-8B"
    chunk_size = 500

    print(f"Starting Streaming Embedding for {metadata.short_name}...", flush=True)
    print(
        f"Workers: {num_processes} | Output: Full 4096 dim (1024 dim saved automatically)"
    )

    start = time.time()

    model_args = (model_name, cache_folder)
    work_generator = chunk_generator(
        data_iterable, metadata, chunk_size, model_args, ground_truths_list
    )

    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=num_processes) as pool:
        results_iter = pool.imap(embed_worker, work_generator)

        all_embeddings = []

        # --- NEW: PROGRESS BAR ---
        # We use tqdm to wrap the iteration.
        # unit="docs" makes it clear we are counting documents, not batches.
        with tqdm(total=total_samples, unit="docs", desc="Embedding") as pbar:
            for batch_emb in results_iter:
                all_embeddings.append(batch_emb)
                # Update progress bar by the actual number of docs in this batch
                pbar.update(len(batch_emb))

    if not all_embeddings:
        print("\nWarning: No data found in dataset!")
        return np.array([])

    final_embeddings = np.concatenate(all_embeddings, axis=0)

    print(f"\nEmbedding finished. Shape: {final_embeddings.shape}")
    print(f"Time taken: {time.time() - start:.2f}s")

    return final_embeddings


def process_dataset(dataset_input, model_dir, output_dir, num_workers):
    # 1. Resolve Name
    if dataset_input not in dataset_info:
        print(f"Dataset '{dataset_input}' not found in dataset_info.", flush=True)
        return
    dataset_key, metadata = dataset_info[dataset_input]
    print(f"Mapped '{dataset_input}' -> '{dataset_key}'", flush=True)

    # 2. Load Data (Streaming)
    load_args = {"path": dataset_key, "cache_dir": model_dir, "streaming": True}
    if metadata.name:
        load_args["name"] = metadata.name

    dataset_obj = datasets.load_dataset(**load_args)
    dataset_iterable = dataset_obj[metadata.split]

    # --- NEW: Try to fetch total sample count for progress bar ---
    total_samples = None
    try:
        # Even in streaming mode, HuggingFace often provides the total count in metadata
        total_samples = (
            dataset_obj[metadata.split].info.splits[metadata.split].num_examples
        )
        print(f"Total documents to process: {total_samples}", flush=True)
    except Exception:
        print(
            "Total document count unknown (streaming). Progress bar will show rate only.",
            flush=True,
        )

    # 3. Embed (Gets full 4096 dims)
    ground_truths = []

    embeddings_full = embed_qwen3_8B(
        dataset_iterable,
        metadata,
        cache_folder=model_dir,
        ground_truths_list=ground_truths,
        num_processes=num_workers,
        total_samples=total_samples,  # <--- PASS TOTAL
    )

    # 4. Generate Matryoshka 1024
    print("Generating Matryoshka 1024-dim version...", flush=True)
    embeddings_1024 = embeddings_full[:, :1024]

    # Re-normalize (Critical)
    norms = np.linalg.norm(embeddings_1024, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    embeddings_1024 = embeddings_1024 / norms

    # 5. Factorize Labels (Maps to 0...k-1)
    print(f"Processing {len(ground_truths)} labels...", flush=True)
    gt_factors = pd.factorize(ground_truths)[0]

    # 6. Save Files
    final_output_dir = os.path.join(output_dir, metadata.short_name)
    Path(final_output_dir).mkdir(parents=True, exist_ok=True)

    # Paths
    path_full = os.path.join(final_output_dir, f"{metadata.short_name}.npy")
    path_1024 = os.path.join(final_output_dir, f"{metadata.short_name}_1024.npy")
    path_gt = os.path.join(final_output_dir, f"{metadata.short_name}.gt")

    # Save Full
    np.save(path_full, embeddings_full)
    print(f"Saved Full (4096): {path_full}")

    # Save 1024
    np.save(path_1024, embeddings_1024)
    print(f"Saved Matryoshka (1024): {path_1024}")

    # Save GT
    with open(path_gt, "w") as f:
        for g in gt_factors:
            f.write(f"{g} \n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate embeddings with Qwen3-8B")

    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name (e.g. 'builtbench', 'reddit') or full MTEB key",
    )

    default_cache = os.path.join(os.path.expanduser("~"), ".cache")
    parser.add_argument(
        "--model_dir", default=default_cache, help="Path to cache/download models"
    )

    parser.add_argument(
        "--output", required=True, help="Output directory for embeddings"
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="Number of worker processes (default: 16)",
    )

    args = parser.parse_args()

    # --- DYNAMIC HARDWARE OPTIMIZATION ---
    total_physical_cores = 192
    # threads_per_worker = 16  # max(1, total_physical_cores // args.num_workers)

    # os.environ["OMP_NUM_THREADS"] = str(threads_per_worker)
    # os.environ["MKL_NUM_THREADS"] = str(threads_per_worker)
    # os.environ["OPENBLAS_NUM_THREADS"] = str(threads_per_worker)

    # print(
    #     f"Configuration: {args.num_workers} Workers | ~{threads_per_worker} Threads per Worker (Total Cores: {total_physical_cores})"
    # )
    print(
        f"Configuration: {args.num_workers} Workers (Total Cores: {total_physical_cores})"
    )

    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass

    process_dataset(args.dataset, args.model_dir, args.output, args.num_workers)
