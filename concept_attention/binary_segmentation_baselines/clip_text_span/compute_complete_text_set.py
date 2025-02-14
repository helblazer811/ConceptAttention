import time
import numpy as np
import torch
from PIL import Image
import glob
import sys
import os
import einops
from torch.utils.data import DataLoader
import tqdm
import argparse
from torchvision.datasets import ImageNet
from pathlib import Path

from clip_text_span.utils.misc import accuracy


@torch.no_grad()
def replace_with_iterative_removal(data, text_features, texts, iters, rank, device):
    results = []
    u, s, vh = np.linalg.svd(data, full_matrices=False)
    vh = vh[:rank]
    text_features = (
        vh.T.dot(np.linalg.inv(vh.dot(vh.T)).dot(vh)).dot(text_features.T).T
    )  # Project the text to the span of W_OV
    data = torch.from_numpy(data).float().to(device)
    mean_data = data.mean(dim=0, keepdim=True)
    data = data - mean_data
    reconstruct = einops.repeat(mean_data, "A B -> (C A) B", C=data.shape[0])
    reconstruct = reconstruct.detach().cpu().numpy()
    text_features = torch.from_numpy(text_features).float().to(device)
    for i in range(iters):
        projection = data @ text_features.T
        projection_std = projection.std(axis=0).detach().cpu().numpy()
        top_n = np.argmax(projection_std)
        results.append(texts[top_n])
        text_norm = text_features[top_n] @ text_features[top_n].T
        reconstruct += (
            (
                (data @ text_features[top_n] / text_norm)[:, np.newaxis]
                * text_features[top_n][np.newaxis, :]
            )
            .detach()
            .cpu()
            .numpy()
        )
        data = data - (
            (data @ text_features[top_n] / text_norm)[:, np.newaxis]
            * text_features[top_n][np.newaxis, :]
        )
        text_features = (
            text_features
            - (text_features @ text_features[top_n] / text_norm)[:, np.newaxis]
            * text_features[top_n][np.newaxis, :]
        )
    return reconstruct, results


def get_args_parser():
    parser = argparse.ArgumentParser("Completeness part", add_help=False)

    # Model parameters
    parser.add_argument(
        "--model",
        default="ViT-H-14",
        type=str,
        metavar="MODEL",
        help="Name of model to use",
    )
    # Dataset parameters
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument(
        "--output_dir", default="./output_dir", help="path where data is saved"
    )
    parser.add_argument(
        "--input_dir", default="./output_dir", help="path where data is saved"
    )
    parser.add_argument(
        "--text_descriptions",
        default="image_descriptions_per_class",
        type=str,
        help="name of the evalauted text set",
    )
    parser.add_argument(
        "--text_dir",
        default="./text_descriptions",
        type=str,
        help="The folder with the text files",
    )
    parser.add_argument(
        "--dataset", type=str, default="imagenet", help="imagenet or waterbirds"
    )
    parser.add_argument(
        "--num_of_last_layers",
        type=int,
        default=8,
        help="How many attention layers to replace.",
    )
    parser.add_argument(
        "--w_ov_rank", type=int, default=80, help="The rank of the OV matrix"
    )
    parser.add_argument(
        "--texts_per_head",
        type=int,
        default=10,
        help="The number of text examples per head.",
    )
    parser.add_argument("--device", default="cuda:0", help="device to use for testing")
    return parser


def main(args):
    with open(
        os.path.join(args.input_dir, f"{args.dataset}_attn_{args.model}.npy"), "rb"
    ) as f:
        attns = np.load(f)  # [b, l, h, d]
    with open(
        os.path.join(args.input_dir, f"{args.dataset}_mlp_{args.model}.npy"), "rb"
    ) as f:
        mlps = np.load(f)  # [b, l+1, d]
    with open(
        os.path.join(args.input_dir, f"{args.dataset}_classifier_{args.model}.npy"),
        "rb",
    ) as f:
        classifier = np.load(f)
    print(f"Number of layers: {attns.shape[1]}")
    all_images = set()
    # Mean-ablate the other parts
    for i in tqdm.trange(attns.shape[1] - args.num_of_last_layers):
        for head in range(attns.shape[2]):
            attns[:, i, head] = np.mean(attns[:, i, head], axis=0, keepdims=True)
    # Load text:
    with open(
        os.path.join(args.input_dir, f"{args.text_descriptions}_{args.model}.npy"), "rb"
    ) as f:
        text_features = np.load(f)
    with open(os.path.join(args.text_dir, f"{args.text_descriptions}.txt"), "r") as f:
        lines = [i.replace("\n", "") for i in f.readlines()]
    with open(
        os.path.join(
            args.output_dir,
            f"{args.dataset}_completeness_{args.text_descriptions}_top_{args.texts_per_head}_heads_{args.model}.txt",
        ),
        "w",
    ) as w:
        for i in tqdm.trange(attns.shape[1] - args.num_of_last_layers, attns.shape[1]):
            for head in range(attns.shape[2]):
                results, images = replace_with_iterative_removal(
                    attns[:, i, head],
                    text_features,
                    lines,
                    args.texts_per_head,
                    args.w_ov_rank,
                    args.device,
                )
                attns[:, i, head] = results
                all_images |= set(images)
                w.write(f"------------------\n")
                w.write(f"Layer {i}, Head {head}\n")
                w.write(f"------------------\n")
                for text in images:
                    w.write(f"{text}\n")

        mean_ablated_and_replaced = mlps.sum(axis=1) + attns.sum(axis=(1, 2))
        projections = torch.from_numpy(mean_ablated_and_replaced).float().to(
            args.device
        ) @ torch.from_numpy(classifier).float().to(args.device)
        labels = np.array([i // 50 for i in range(attns.shape[0])])
        current_accuracy = (
            accuracy(projections.cpu(), torch.from_numpy(labels))[0] * 100.0
        )
        print(
            f"Current accuracy:",
            current_accuracy,
            "\nNumber of texts:",
            len(all_images),
        )
        w.write(f"------------------\n")
        w.write(
            f"Current accuracy: {current_accuracy}\nNumber of texts: {len(all_images)}"
        )


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
