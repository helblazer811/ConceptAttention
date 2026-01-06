"""
    Test the diffusion scope approach on different timesteps for the DiT.

    For each of 10 different timesteps in the DiT:
    
    1. Run a forward pass for the model. 
    2. Compute pixelwise average, mIoU, and AP for each layer of the model for the given example.
    3. Save the metrics for each layer for each example in a CSV. 

    Plots:
    1. Single line chart for pixelwise average, mIoU, and AP for each layer.
    2. Three line charts, one for each metric.
"""
import pandas as pd
from torchvision import transforms 
from PIL import Image
import argparse
import numpy as np
import torch
import os
import einops
from tqdm import tqdm
import matplotlib.pyplot as plt 

from experiments.paper_figures.imagenet_segmentation.data_processing import ImagenetSegmentation
from concept_attention.flux.image_generator import FluxGenerator
from concept_attention.segmentation import generate_concept_basis_and_image_representation
from concept_attention.utils import batch_intersection_union, batch_pix_accuracy, get_ap_scores

if __name__ == "__main__":
    # Parse the arguments
    parser = argparse.ArgumentParser()
    # Arguments
    parser.add_argument("--device", type=str, default="cuda:2", help="The device to run the model on.")
    parser.add_argument("--model_name", type=str, default="flux-dev", help="The model name to use.")
    parser.add_argument("--offload", type=bool, default=False, help="Whether to offload the model to the device.")
    parser.add_argument("--num_steps", type=int, default=50, help="The number of steps to run the model for.")
    parser.add_argument("--seed", type=int, default=1, help="The seed to use for the model.")
    parser.add_argument("--target_space", type=str, default="output", help="The target space to use for the model.")
    parser.add_argument("--height", type=int, default=1024, help="The height of the image.")
    parser.add_argument("--width", type=int, default=1024, help="The width of the image.")
    parser.add_argument("--num_samples", type=int, default=1, help="The number of samples to use for the model.")
    parser.add_argument("--joint_attention_kwargs", type=dict, default=None, help="The joint attention arguments to use for the model.")

    args = parser.parse_args()

    # Load up the flux model
    flux_generator = FluxGenerator(
        model_name=args.model_name,
        device=args.device,
        offload=args.offload,
    )
    # Load up the imagenet dataset
    dataset = ImagenetSegmentation(
        directory=os.path.join("..", "imagenet_segmentation", "data", "imagenet_segmentation"),
    )
    # Make transforms
    image_transforms = transforms.Compose([
        transforms.Resize((args.width, args.height), Image.NEAREST),
        transforms.ToTensor(),
    ])
    label_transforms = transforms.Compose([
        transforms.Resize((224, 224), Image.NEAREST),
        transforms.ToTensor()
    ])
    # Make a pandas df 
    df = pd.DataFrame(columns=["image_id", "timestep", "pixelwise_average", "mIoU", "AP"])
    # Iterate through the examples
    for index in tqdm(range(len(dataset))):
        img, labels, simplified_name = dataset[index]
        # Apply transformations
        image = image_transforms(img)
        labels = label_transforms(labels)
        # Iterate through each timestep
        for timestep in range(args.num_steps):
            # Run the segmentation model
            image_vectors, concept_vectors, reconstructed_image = generate_concept_basis_and_image_representation(
                image=image,
                caption=f"a {simplified_name}",
                concepts=[
                    simplified_name, 
                    "background"
                    "floor",
                    "tree",
                    "person",
                    # "grass",
                    "face"
                ],
                noise_timestep=timestep,
                layers=list(range(19)),
                normalize_concepts=True,
                num_steps=args.num_steps,
                seed=args.seed,
                model_name=args.model_name,
                offload=args.offload,
                device=args.device,
                target_space=args.target_space,
                height=args.height,
                width=args.width,
                generator=flux_generator,
                stop_after_multimodal_attentions=True,
                num_samples=args.num_samples,
                joint_attention_kwargs=None
            )
            # Plot concept heatmaps
            coefficients = einops.einsum(
                concept_vectors,
                image_vectors,
                "concepts dim, patches dim -> concepts patches",
            )
            # Create a mask from the coefficients
            coefficients = coefficients[0] # Pull out just the foreground mask
            coefficients = einops.rearrange(
                coefficients,
                "(h w) -> h w",
                w=64,
                h=64
            )
            plt.figure(figsize=(10, 10))
            plt.imshow(coefficients.detach().cpu().numpy())
            plt.axis("off")
            plt.savefig("results/coefficients.png")
            mask = torch.zeros_like(coefficients)
            mean_value = coefficients.mean()
            mask[coefficients > mean_value] = 1
            # Upscale back to 224 x 224
            coefficients = torch.nn.functional.interpolate(
                coefficients.unsqueeze(0).unsqueeze(0),
                size=(224, 224),
                mode="bilinear"
            ).squeeze()
            mask = torch.nn.functional.interpolate(
                mask.unsqueeze(0).unsqueeze(0),
                size=(224, 224),
                mode="nearest"
            ).squeeze()
            # c. Evaluate the pixelwise average, mIoU, and AP for each layer.
            labels = labels.bool().detach().cpu().numpy().squeeze()
            unpadded_target = torch.Tensor(labels)
            unpadded_coefficients = torch.Tensor(coefficients)
            unpadded_mask = torch.stack((1 - mask, mask))
            unpadded_target = torch.stack((1 - unpadded_target, unpadded_target))
            current_correct, current_label = batch_pix_accuracy(unpadded_mask, unpadded_target) # (batch_size, h * w)
            # Add a 1 - mask and 1 - label
            current_inter, current_union = batch_intersection_union(unpadded_mask, unpadded_target, nclass=2)
            unpadded_coefficients = torch.stack((1 - unpadded_coefficients, unpadded_coefficients)).unsqueeze(0)
            labels = torch.Tensor(labels).unsqueeze(0)
            ap_score = np.nan_to_num(
                get_ap_scores(unpadded_coefficients, labels)
            )
            pixAcc = (
                np.float64(1.0)
                * current_correct
                / (np.spacing(1, dtype=np.float64) + current_label)
            )
            IoU = (
                np.float64(1.0)
                * current_inter
                / (np.spacing(1, dtype=np.float64) + current_union)
            )
            mIoU = IoU.mean()
            # d. Save the necessary information in a CSV: [image_id, layer, pixelwise_average, mIoU, AP]
            df = pd.concat([
                    df,
                    pd.DataFrame({
                        "image_id": index,
                        "timestep": timestep,
                        "pixelwise_average": pixAcc,
                        "mIoU": mIoU,
                        "AP": ap_score
                    })
                ], 
                ignore_index=True
            )
            # Save the df
            df.to_csv("results/per_time_metrics.csv")