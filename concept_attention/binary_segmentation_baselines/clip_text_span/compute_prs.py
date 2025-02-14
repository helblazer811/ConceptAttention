import numpy as np
import torch
from PIL import Image
import os.path
import argparse
from pathlib import Path

from torch.utils.data import DataLoader
import tqdm
from concept_attention.binary_segmentation_baselines.clip_text_span.utils.factory import create_model_and_transforms, get_tokenizer
from concept_attention.binary_segmentation_baselines.clip_text_span.utils.binary_waterbirds import BinaryWaterbirds
from prs_hook import hook_prs_logger
from torchvision.datasets import CIFAR100, CIFAR10, ImageNet, ImageFolder


def get_args_parser():
    parser = argparse.ArgumentParser("Project Residual Stream", add_help=False)
    parser.add_argument("--batch_size", default=2, type=int, help="Batch size")
    # Model parameters
    parser.add_argument(
        "--model",
        default="ViT-H-14",
        type=str,
        metavar="MODEL",
        help="Name of model to use",
    )
    parser.add_argument("--pretrained", default="laion2b_s32b_b79k", type=str)
    # Dataset parameters
    parser.add_argument(
        "--data_path", default="/shared/group/ilsvrc", type=str, help="dataset path"
    )
    parser.add_argument(
        "--dataset", type=str, default="imagenet", help="imagenet, cub or waterbirds"
    )
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument(
        "--output_dir", default="./output_dir", help="path where to save"
    )
    parser.add_argument("--device", default="cuda:0", help="device to use for testing")
    return parser


def main(args):
    """Calculates the projected residual stream for a dataset."""
    model, _, preprocess = create_model_and_transforms(
        args.model, pretrained=args.pretrained
    )
    model.to(args.device)
    model.eval()
    context_length = model.context_length
    vocab_size = model.vocab_size

    print(
        "Model parameters:",
        f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}",
    )
    print("Context length:", context_length)
    print("Vocab size:", vocab_size)
    print("Len of res:", len(model.visual.transformer.resblocks))

    prs = hook_prs_logger(model, args.device)

    # Data:
    if args.dataset == "imagenet":
        ds = ImageNet(root=args.data_path, split="val", transform=preprocess)
    elif args.dataset == "binary_waterbirds":
        ds = BinaryWaterbirds(root=args.data_path, split="test", transform=preprocess)
    elif args.dataset == "CIFAR100":
        ds = CIFAR100(
            root=args.data_path, download=True, train=False, transform=preprocess
        )
    elif args.dataset == "CIFAR10":
        ds = CIFAR10(
            root=args.data_path, download=True, train=False, transform=preprocess
        )
    else:
        ds = ImageFolder(root=args.data_path, transform=preprocess)
    dataloader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    attention_results = []
    mlp_results = []
    cls_to_cls_results = []
    for i, (image, _) in enumerate(tqdm.tqdm(dataloader)):
        with torch.no_grad():
            prs.reinit()
            representation = model.encode_image(
                image.to(args.device), attn_method="head", normalize=False
            )
            attentions, mlps = prs.finalize(representation)
            attentions = attentions.detach().cpu().numpy()  # [b, l, n, h, d]
            mlps = mlps.detach().cpu().numpy()  # [b, l+1, d]
            attention_results.append(
                np.sum(attentions, axis=2)
            )  # Reduce the spatial dimension
            mlp_results.append(mlps)
            cls_to_cls_results.append(
                np.sum(attentions[:, :, 0], axis=2)
            )  # Store the cls->cls attention, reduce the heads
    with open(
        os.path.join(args.output_dir, f"{args.dataset}_attn_{args.model}.npy"), "wb"
    ) as f:
        np.save(f, np.concatenate(attention_results, axis=0))
    with open(
        os.path.join(args.output_dir, f"{args.dataset}_mlp_{args.model}.npy"), "wb"
    ) as f:
        np.save(f, np.concatenate(mlp_results, axis=0))
    with open(
        os.path.join(args.output_dir, f"{args.dataset}_cls_attn_{args.model}.npy"), "wb"
    ) as f:
        np.save(f, np.concatenate(cls_to_cls_results, axis=0))


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
