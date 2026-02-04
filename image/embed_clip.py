import argparse
import os

import clip
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import datasets
from tqdm import tqdm

# Standardize threading for your 96-core server
torch.set_num_threads(os.cpu_count())

# --- CLASSIFICATION DATASET REGISTRY ---
# Format: (DatasetClass, split_logic_type)
dataset_info = {
    # Boolean 'train' flag
    "cifar-10": (datasets.CIFAR10, "train_bool"),
    "cifar-100": (datasets.CIFAR100, "train_bool"),
    "mnist": (datasets.MNIST, "train_bool"),
    "fashion_mnist": (datasets.FashionMNIST, "train_bool"),
    "kmnist": (datasets.KMNIST, "train_bool"),
    "emnist": (
        datasets.EMNIST,
        "train_bool",
    ),  # Needs 'split' arg (e.g., split='balanced')
    # String 'split' argument
    "stl10": (datasets.STL10, "split_str"),
    "svhn": (datasets.SVHN, "split_str"),
    "celeba": (datasets.CelebA, "split_str"),
    "imagenet": (datasets.ImageNet, "split_str"),
    "places365": (datasets.Places365, "split_str"),
    "food101": (datasets.Food101, "split_str_alt"),  # Uses 'split' but only train/test
    "flowers102": (datasets.Flowers102, "split_str_alt"),
    # String 'image_set' argument
    "voc-cls": (datasets.VOCDetection, "image_set"),
}


def get_classification_dataset(dataset_tag, root, transform):
    if dataset_tag not in dataset_info:
        raise ValueError(f"Dataset {dataset_tag} not in registry.")

    DS_Class, logic = dataset_info[dataset_tag]

    # Standardizing the split loading to get the "full" pool for retrieval research
    if logic == "train_bool":
        train = DS_Class(root=root, train=True, download=True, transform=transform)
        test = DS_Class(root=root, train=False, download=True, transform=transform)
    elif logic == "split_str":
        train = DS_Class(root=root, split="train", download=True, transform=transform)
        test = DS_Class(root=root, split="test", download=True, transform=transform)
    elif logic == "split_str_alt":
        # Some use 'train' and 'test', others 'train' and 'val'
        train = DS_Class(root=root, split="train", download=True, transform=transform)
        test = DS_Class(
            root=root,
            split="test" if dataset_tag != "flowers102" else "test",
            download=True,
            transform=transform,
        )
    elif logic == "image_set":
        train = DS_Class(
            root=root, image_set="train", download=True, transform=transform
        )
        test = DS_Class(root=root, image_set="val", download=True, transform=transform)

    return ConcatDataset([train, test])


def run_embedding(dataset_tag, model_dir, output_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    dataset = get_classification_dataset(dataset_tag, model_dir, preprocess)

    # 96-core optimization: Use high num_workers for image decoding
    loader = DataLoader(
        dataset,
        batch_size=1024,
        shuffle=False,
        num_workers=16,
        pin_memory=True if device == "cuda" else False,
    )

    all_embeddings, all_labels = [], []
    print(f"Generating CLIP embeddings for {dataset_tag}...")

    with torch.no_grad():
        for images, labels in tqdm(loader):
            images = images.to(device)
            features = model.encode_image(images).float()
            features /= features.norm(dim=-1, keepdim=True)

            all_embeddings.append(features.cpu().numpy())
            all_labels.extend(labels.tolist() if torch.is_tensor(labels) else labels)

    # Save logic
    os.makedirs(output_path, exist_ok=True)
    np.save(
        os.path.join(output_path, f"{dataset_tag}.npy"),
        np.concatenate(all_embeddings),
    )

    with open(os.path.join(output_path, f"{dataset_tag}.gt"), "w") as f:
        f.write("\n".join(map(str, all_labels)) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model_dir", default=os.path.expanduser("~/.cache"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    run_embedding(args.dataset, args.model_dir, args.output)
