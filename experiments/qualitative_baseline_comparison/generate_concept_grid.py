import argparse
from PIL import Image
import einops
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms

from concept_attention.binary_segmentation_baselines.clip_text_span_baseline import CLIPTextSpanSegmentationModel
from concept_attention.binary_segmentation_baselines.daam_sdxl import DAAMStableDiffusionXLSegmentationModel
from concept_attention.binary_segmentation_baselines.raw_cross_attention import RawCrossAttentionSegmentationModel
from concept_attention.binary_segmentation_baselines.raw_output_space import RawOutputSpaceSegmentationModel
from concept_attention.flux.image_generator import FluxGenerator

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generate grid of images')

    # parser.add_argument('--prompt', type=str, default="A centered fire breathing dragon in the sky. Sun in the background.")
    # parser.add_argument('--concepts', type=list, default=["dragon", "flames", "sky", "sun", "rock"])
    # parser.add_argument('--seed', type=int, default=6)
    # parser.add_argument('--prompt', type=str, default="A fire breathing green dragon in the sunny sky.")
    # parser.add_argument('--concepts', type=list, default=["dragon", "flames", "sky", "rock", "sun"])
    parser.add_argument('--prompt', type=str, default="A dragon on a rock with wings spread. Sun in sky. .")
    parser.add_argument('--concepts', type=list, default=["rock", "dragon", "sky", "cloud", "sun"])
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--device', type=str, default="cuda:1")
    parser.add_argument('--layers', type=list, default=list(range(10, 19)))
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--method_names', type=list, default=["RawOutputSpace", "RawCrossAttention", "DAAMSDXL", "TextSpan"])

    args = parser.parse_args()

    # 1. Generate the image
    generator = FluxGenerator(
        model_name="flux-schnell",
        device=args.device,
        offload=False,
    )
    image = generator.generate_image(
        height=1024,
        width=1024,
        num_steps=4,
        guidance=0.0,
        prompt=args.prompt,
        concepts=args.concepts,
        seed=args.seed
    )
    image.save("results/image.png")
    # Transform the image
    image_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    image = image_transform(image)
    # Delete the generator
    del generator # Free up memory
    # Make the figure
    fig, axs = plt.subplots(len(args.method_names), len(args.concepts), figsize=(4 * (len(args.concepts)), 4 * len(args.method_names)))
    # 2. Iterate through each method (lazy loading the model to save memory)
    segmentation_model = None
    for method_index, method in enumerate(args.method_names):
        del segmentation_model
        # Load the model
        if method == "RawCrossAttention":
            segmentation_model = RawCrossAttentionSegmentationModel(
                model_name="flux-schnell",
                device=args.device,
                offload=False,
            )
        elif method == "RawOutputSpace":
            segmentation_model = RawOutputSpaceSegmentationModel(
                model_name="flux-schnell",
                device=args.device,
                offload=False,
            )
        elif method == "TextSpan":
            segmentation_model = CLIPTextSpanSegmentationModel(device=args.device)
        elif method == "DAAMSDXL":
            segmentation_model = DAAMStableDiffusionXLSegmentationModel(device=args.device)
        # Do the segmentation
        coefficients, _ = segmentation_model.segment_individual_image(
            image,
            args.concepts,
            caption=args.prompt,
            device=args.device,
            softmax=True,
            layers=args.layers,
            num_samples=args.num_samples
        )
        # Plot the coefficients
        vmin = coefficients.min()
        vmax = coefficients.max()
        # Save each coefficient map as an image
        for concept_index, concept in enumerate(args.concepts):
            if method_index == 0:
                axs[method_index, concept_index].set_title(concept)
            concept_heatmap = coefficients[concept_index].cpu().numpy()
            axs[method_index, concept_index].imshow(concept_heatmap, cmap="plasma", vmin=vmin, vmax=vmax)
            axs[method_index, concept_index].axis("off")
            # axs[method_index, concept_index].set_title(f"{method} - {concept}")
            # plt.figure(figsize=(10, 10))
            # plt.imshow(concept_heatmap, cmap="plasma", vmin=vmin, vmax=vmax)
            # plt.axis("off")
            # plt.tight_layout()
            # plt.savefig(f"results/{method}/{concept}.png")
            # plt.close()

    plt.tight_layout()
    plt.savefig(f"results/concept_grid.png")