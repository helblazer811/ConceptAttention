import time
import numpy as np
import torch
from PIL import Image
import glob
import sys
import os.path
import argparse
import datetime
import json
from pathlib import Path
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import tqdm

from concept_attention.binary_segmentation_baselines.clip_text_span.utils.factory import create_model_and_transforms, get_tokenizer


def get_args_parser():
    parser = argparse.ArgumentParser('Get text list weights', add_help=False)
    # Model parameters
    parser.add_argument('--batch_size', default=2048, type=int,
                        help='Batch size')
    parser.add_argument('--model', default='ViT-H-14', type=str, metavar='MODEL',
                        help='Name of model to use')
    parser.add_argument('--pretrained', default='laion2b_s32b_b79k', type=str)
    # Dataset parameters
    parser.add_argument('--data_path', default='text_descriptions/image_descriptions_general.txt', 
                        type=str, help='dataset path')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save')
    parser.add_argument('--device', default='cuda:0',
                        help='device to use for testing')
    return parser



def get_text_features(model, tokenizer, lines, 
                      device, batch_size, amp=True, use_format=False):
    """
    This function returns zero-shot vectors for each class in order
    to use it for zero-shot classification.
    

    model:
        CLIP-like model with `encode_text`
    
    tokenizer:
        text tokenizer, i.e. convert list of strings to torch.Tensor of integers
    
    lines: list of str
        name of classes
    
    Returns
    -------
    
    torch.Tensor of shape (N,C) where N is the number
    of templates, and C is the number of classes.
    """
    autocast = torch.cuda.amp.autocast
    with torch.no_grad(), autocast():
        zeroshot_weights = []
        for i in tqdm.trange(0, len(lines), batch_size):
            texts = lines[i:i+batch_size]
            texts = tokenizer(texts).to(device)  # tokenize
            class_embeddings = model.encode_text(texts)
            class_embeddings = F.normalize(class_embeddings, dim=-1)
            zeroshot_weights.append(class_embeddings.detach().cpu())
        zeroshot_weights = torch.concatenate(zeroshot_weights, dim=0)
    return zeroshot_weights


def main(args):
    """Calculates the classifier projection weights."""
    model, _, preprocess = create_model_and_transforms(args.model, pretrained=args.pretrained)
    tokenizer = get_tokenizer(args.model)
    model.to(args.device)
    model.eval()
    context_length = model.context_length
    vocab_size = model.vocab_size

    print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
    print("Context length:", context_length)
    print("Vocab size:", vocab_size)
    with open(args.data_path, 'r') as f:
        lines = f.readlines()
    base, name = os.path.split(args.data_path)
    name = name.replace('.txt', '')
    features = get_text_features(model, tokenizer, lines, args.device, args.batch_size)
    with open(os.path.join(args.output_dir, f'{name}_{args.model}.npy'), 'wb') as f:
        np.save(f, features.numpy())
    
    
if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)