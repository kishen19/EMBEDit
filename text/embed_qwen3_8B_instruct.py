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

# --- OFFICIAL QWEN3 INSTRUCTION REGISTRY ---
# Mapped from the official JSON provided by the user.
# Optimized the TREC-COVID instruction to be topic-specific rather than generic.
QWEN_INSTRUCTIONS = {
    # ArXiv
    "arxiv": "Identify the main and secondary category of Arxiv papers based on the titles",
    "arxiv_p2p": "Identify the main and secondary category of Arxiv papers based on the titles and abstracts",
    # BioRxiv / MedRxiv
    "biorxiv": "Identify the main category of Biorxiv papers based on the titles",
    "biorxiv_p2p": "Identify the main category of Biorxiv papers based on the titles and abstracts",
    "medrxiv": "Identify the main category of Medrxiv papers based on the titles",
    "medrxiv_p2p": "Identify the main category of Medrxiv papers based on the titles and abstracts",
    # TREC-COVID (OPTIMIZED)
    # The labels are granular topics like "origin", "incubation period", etc.
    "trec_covid": "Identify the specific coronavirus research topic or question that this document answers",
    "trec_covid_p2p": "Identify the specific coronavirus research topic or question that this document answers",
    # Reddit / Social
    "reddit": "Identify the topic or theme of Reddit posts based on the titles",
    "reddit_p2p": "Identify the topic or theme of Reddit posts based on the titles and posts",
    # StackExchange
    "stackexchange": "Identify the topic or theme of StackExchange posts based on the titles",
    "stackexchange_p2p": "Identify the topic or theme of StackExchange posts based on the given paragraphs",
    # News / Wikipedia / General
    "twentynewsgroups": "Identify the topic or theme of the given news articles",
    "wikipedia_medium": "Identify the category of wiki passages",
    "wikipedia_easy": "Identify the category of wiki passages",
    "cities_wiki": "Identify of Wikipedia articles of cities by country",
    # Patents / Technical
    "big_patent": "Identify the category of documents from the Big Patent dataset",
    "builtbench": "Identify the topic or theme of the given technical benchmark descriptions",
    "builtbench_p2p": "Identify the topic or theme of the given technical benchmark descriptions",
    # AlloProf (Education)
    "alloprof": "Identify the topic of document titles from Allo Prof dataset",
    "alloprof_p2p": "Identify the main category of Allo Prof document based on the titles and descriptions",
    # HAL (Academic)
    "hal": "Identify the main category of academic passage based on the titles and contents",
}

# Fallback instruction if a key is missed
DEFAULT_INSTRUCTION = "Identify the topic or theme of the given text"


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
        Metadata("alloprof", "test", "sentences", "labels", False, language="fra"),
    ),
    "alloprof_p2p": (
        "mteb/AlloProfClusteringP2P.v2",
        Metadata("alloprof_p2p", "test", "sentences", "labels", False, language="fra"),
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
        Metadata(
            "hal", "test", "title", "domain", False, name="mteb_eval", language="fra"
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


def format_qwen_input(instruction, text):
    """
    Applies the official Qwen3 instruction format:
    Instruct: {instruction}\nQuery: {text}
    """
    return f"Instruct: {instruction}\nQuery: {text}"


def process_dataset(dataset_input, model_dir, output_dir):
    if dataset_input not in dataset_info:
        print(f"Dataset '{dataset_input}' not found.")
        return

    dataset_key, metadata = dataset_info[dataset_input]

    # Retrieve the specific instruction for this dataset
    instruction = QWEN_INSTRUCTIONS.get(dataset_input, DEFAULT_INSTRUCTION)
    print(f'Using Instruction for {dataset_input}: "{instruction}"')

    # 1. Initialize Model
    has_gpu = torch.cuda.is_available()
    model_name = "Qwen/Qwen3-Embedding-8B"
    torch_dtype = "float16" if has_gpu else "float32"

    model = SentenceTransformer(
        model_name,
        cache_folder=model_dir,
        trust_remote_code=True,
        model_kwargs={"torch_dtype": torch_dtype},
    )

    # Set context length to avoid truncation
    model.max_seq_length = 4096
    print(
        f"Model loaded. Max Seq Length: {model.max_seq_length} | Precision: {torch_dtype}"
    )

    # 2. Data Preparation
    ds = datasets.load_dataset(dataset_key, name=metadata.name, cache_dir=model_dir)
    split = metadata.split if metadata.split in ds else list(ds.keys())[0]
    data = ds[split]

    texts, labels = [], []

    print(f"Formatting data for {dataset_input}...")
    for item in tqdm(data, desc="Preprocessing"):
        raw_val, l = item[metadata.text_col], item[metadata.label_col]

        # Handle multiple strings per entry
        current_batch = raw_val if metadata.multiple_per_entry else [raw_val]

        processed_batch = []
        for text in current_batch:
            # Apply strict Qwen formatting: "Instruct: ... \nQuery: ..."
            formatted_text = format_qwen_input(instruction, str(text))
            processed_batch.append(formatted_text)

        texts.extend(processed_batch)
        labels.extend(l if metadata.multiple_per_entry else [l])

    # 3. Multi-Process Execution
    # 96-core optimization: Use 8 processes to balance memory and CPU throughput
    target_devices = None if has_gpu else ["cpu"] * 8
    print(f"Starting multi-process pool on {'GPU' if has_gpu else 'CPU'}...")
    pool = model.start_multi_process_pool(target_devices=target_devices)

    start_time = time.time()
    embeddings_full = model.encode_multi_process(
        texts, pool, batch_size=128 if has_gpu else 8, chunk_size=500
    )
    model.stop_multi_process_pool(pool)
    print(f"Embedding finished in {time.time() - start_time:.2f}s")

    # 4. Save Logic
    save_dir = Path(output_dir) / metadata.short_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Matryoshka Truncation (1024 dims)
    embeddings_1024 = embeddings_full[:, :1024]
    norms = np.linalg.norm(embeddings_1024, axis=1, keepdims=True)
    embeddings_1024 /= np.where(norms == 0, 1e-10, norms)

    np.save(save_dir / f"{metadata.short_name}_instruct.npy", embeddings_full)
    np.save(save_dir / f"{metadata.short_name}_instruct_1024.npy", embeddings_1024)

    gt_factors = pd.factorize(labels)[0]
    with open(save_dir / f"{metadata.short_name}_instruct.gt", "w") as f:
        f.write("\n".join(map(str, gt_factors)) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model_dir", default=os.path.expanduser("~/.cache"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    process_dataset(args.dataset, args.model_dir, args.output)
