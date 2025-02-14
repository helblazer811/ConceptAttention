import numpy as np
import torch
import os.path
import argparse
import einops
from pathlib import Path

import tqdm
from concept_attention.binary_segmentation_baselines.clip_text_span.utils.misc import accuracy


def get_args_parser():
    parser = argparse.ArgumentParser("Ablations part", add_help=False)

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
        "--figures_dir", default="./output_dir", help="path where data is saved"
    )
    parser.add_argument(
        "--input_dir", default="./output_dir", help="path where data is saved"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="imagenet",
        help="imagenet, waterbirds, cub, binary_waterbirds",
    )
    return parser


def main(args):
    
    attns = np.load(os.path.join(args.input_dir, f"{args.dataset}_attn_{args.model}.npy"), mmap_mode="r")  # [b, l, h, d]
    mlps = np.load(os.path.join(args.input_dir, f"{args.dataset}_mlp_{args.model}.npy"), mmap_mode="r")  # [b, l+1, d]
    with open(
        os.path.join(args.input_dir, f"{args.dataset}_classifier_{args.model}.npy"),
        "rb",
    ) as f:
        classifier = np.load(f)
    if args.dataset == "imagenet":
        labels = np.array([i // 50 for i in range(attns.shape[0])])
    else:
        with open(
            os.path.join(args.input_dir, f"{args.dataset}_labels.npy"), "rb"
        ) as f:
            labels = np.load(f)
    baseline = attns.sum(axis=(1, 2)) + mlps.sum(axis=1)
    baseline_acc = (
        accuracy(
            torch.from_numpy(baseline @ classifier).float(), torch.from_numpy(labels)
        )[0]
        * 100
    )
    print("Baseline:", baseline_acc)
    mlps_mean = einops.repeat(mlps.mean(axis=0), "l d -> b l d", b=attns.shape[0])
    mlps_ablation = attns.sum(axis=(1, 2)) + mlps_mean.sum(axis=1)
    mlps_ablation_acc = (
        accuracy(
            torch.from_numpy(mlps_ablation @ classifier).float(),
            torch.from_numpy(labels),
        )[0]
        * 100
    )
    print("+ MLPs ablation:", mlps_ablation_acc)
    mlps_no_layers = mlps.sum(axis=1)
    attns_no_cls = attns.sum(axis=2)
    with open(
        os.path.join(args.input_dir, f"{args.dataset}_cls_attn_{args.model}.npy"), "rb"
    ) as f:
        cls_attn = np.load(f)  # [b, l, d]
    attns_no_cls = attns_no_cls - cls_attn + cls_attn.mean(axis=0)[np.newaxis, :, :]
    no_cls_ablation = attns_no_cls.sum(axis=1) + mlps_no_layers
    no_cls_acc = (
        accuracy(
            torch.from_numpy(no_cls_ablation @ classifier).float(),
            torch.from_numpy(labels),
        )[0]
        * 100
    )
    print("+ CLS ablation:", no_cls_acc)
    mlp_and_no_cls_ablation = attns_no_cls.sum(axis=1) + mlps_mean.sum(axis=1)
    mlp_and_no_cls_ablation_acc = (
        accuracy(
            torch.from_numpy(mlp_and_no_cls_ablation @ classifier).float(),
            torch.from_numpy(labels),
        )[0]
        * 100
    )
    print("+ MLPs + CLS ablation:", mlp_and_no_cls_ablation_acc)
    no_heads_attentions = attns.sum(axis=(2))
    all_accuracies = [baseline_acc]
    for layer in range(attns.shape[1]):
        current_model = (
            np.sum(
                np.mean(no_heads_attentions[:, :layer], axis=0, keepdims=True), axis=1
            )
            + np.mean(no_heads_attentions[:, layer], axis=0, keepdims=True)
            + np.sum(no_heads_attentions[:, layer + 1 :], axis=1)
        )
        current_accuracy = (
            accuracy(
                torch.from_numpy((mlps_no_layers + current_model) @ classifier).float(),
                torch.from_numpy(labels),
            )[0]
            * 100
        )
        all_accuracies.append(current_accuracy)
    print("Attention ablations:", all_accuracies)


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.figures_dir:
        Path(args.figures_dir).mkdir(parents=True, exist_ok=True)
    main(args)
