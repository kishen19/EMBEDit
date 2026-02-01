import argparse
import datasets
import pandas as pd


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


dataset_metadata = {
    # 617, 4
    "BASF-AI/WikipediaMedium5Clustering": Metadata(
        "WikipediaMedium5Clustering", "test", "sentences", "labels", True
    ),
    # 2105, 9
    "BASF-AI/WikipediaEasy10Clustering": Metadata(
        "WikipediaEasy10Clustering", "test", "sentences", "labels", True
    ),
    # 2556, 13
    "mteb/AlloProfClusteringS2S.v2": Metadata(
        "AlloProfClusteringS2S.v2", "test", "sentences", "labels", False, language="fra"
    ),
    # 2556, 13
    "mteb/AlloProfClusteringP2P.v2": Metadata(
        "AlloProfClusteringP2P.v2", "test", "sentences", "labels", False, language="fra"
    ),
    # 2284, 50
    "Uri-ka/ClusTREC-Covid": Metadata(
        "ClusTREC-Covid", "test", "sentences", "labels", False, "title"
    ),
    # 3531, 126
    "jinaai/cities_wiki_clustering": Metadata(
        "cities_wiki_clustering", "test", "sentences", "labels", True
    ),
    # 3815, 31
    "mehrzad-shahin/BuiltBench-clustering-s2s": Metadata(
        "BuiltBench-clustering-s2s", "test", "sentences", "labels", True
    ),
    # 4577, 35
    "mehrzad-shahin/BuiltBench-clustering-p2p": Metadata(
        "BuiltBench-clustering-p2p", "test", "sentences", "labels", True
    ),
    # 17647, 51
    "mteb/medrxiv-clustering-p2p": Metadata(
        "medrxiv-clustering-p2p", "test", "sentences", "labels", False
    ),
    # 17647, 51
    "mteb/medrxiv-clustering-s2s": Metadata(
        "medrxiv-clustering-s2s", "test", "sentences", "labels", False
    ),
    # 26233, 10
    "lyon-nlp/clustering-hal-s2s": Metadata(
        "clustering-hal-s2s",
        "test",
        "title",
        "domain",
        False,
        "mteb_eval",
        language="fra",
    ),
    # 53787, 26
    "mteb/biorxiv-clustering-s2s": Metadata(
        "biorxiv-clustering-s2s", "test", "sentences", "labels", False
    ),
    # 53787, 26
    "mteb/biorxiv-clustering-p2p": Metadata(
        "biorxiv-clustering-p2p", "test", "sentences", "labels", False
    ),
    # 59545, 20
    "mteb/twentynewsgroups-clustering": Metadata(
        "twentynewsgroups-clustering", "test", "sentences", "labels", True
    ),
    # 67066, 9
    "mteb/big-patent": Metadata("big-patent", "test", "sentences", "labels", False),
    # 75000, 610
    "mteb/stackexchange-clustering-p2p": Metadata(
        "stackexchange-clustering-p2p", "test", "sentences", "labels", True
    ),
    # 373850, 121
    "mteb/stackexchange-clustering": Metadata(
        "stackexchange-clustering", "test", "sentences", "labels", True
    ),
    # 420464, 50
    "mteb/reddit-clustering": Metadata(
        "reddit-clustering", "test", "sentences", "labels", True
    ),
    # 459399, 450
    "mteb/reddit-clustering-p2p": Metadata(
        "reddit-clustering-p2p", "test", "sentences", "labels", True
    ),
    # 732723, 180
    "mteb/arxiv-clustering-s2s": Metadata(
        "arxiv-clustering-s2s", "test", "sentences", "labels", True
    ),
    # 732723, 180
    "mteb/arxiv-clustering-p2p": Metadata(
        "arxiv-clustering-p2p", "test", "sentences", "labels", True
    ),
}


def get_dataset_info(dataset_name, metadata):
    """
    Gets the size and number of clusters for a dataset.
    """
    print(f"Processing dataset: {dataset_name}", flush=True)
    # Load the dataset in streaming mode to avoid downloading everything
    if metadata.name:
        data = datasets.load_dataset(
            dataset_name, name=metadata.name, split=metadata.split, streaming=True
        )
    else:
        data = datasets.load_dataset(dataset_name, split=metadata.split, streaming=True)

    num_documents = 0
    all_labels = []

    for document in data:
        labels = document[metadata.label_col]
        texts = document[metadata.text_col]

        if not metadata.multiple_per_entry:
            labels = [labels]
            texts = [texts]

        all_labels.extend(labels)
        num_documents += len(texts)

    num_clusters = len(pd.unique(all_labels))

    print(f"  - Dataset: {metadata.short_name}", flush=True)
    print(f"  - Number of documents: {num_documents}", flush=True)
    print(f"  - Number of clusters: {num_clusters}", flush=True)
    print("-" * 20, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get dataset size and cluster info.")
    parser.add_argument(
        "--dataset_name",
        help="Name of the dataset to process. If not provided, all datasets in dataset_metadata will be processed.",
        default=None,
    )

    args = parser.parse_args()

    if args.dataset_name:
        if args.dataset_name in dataset_metadata:
            get_dataset_info(args.dataset_name, dataset_metadata[args.dataset_name])
        else:
            print(
                f"Dataset '{args.dataset_name}' not found in dataset_metadata.",
                flush=True,
            )
    else:
        for name, meta in dataset_metadata.items():
            get_dataset_info(name, meta)
