"""
    Producing concept heatmaps for a generated image using Stable Diffusion 3.
"""
import os
import torch
import einops
import matplotlib.pyplot as plt

from concept_attention.sd3.pipeline import CustomStableDiffusion3Pipeline
from concept_attention.sd3.custom_mmdit import CustomSD3Transformer2DModel

if __name__ == "__main__":
    # Create output directory
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "results", "sd3")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load the model
    model_id = "stabilityai/stable-diffusion-3.5-large-turbo"
    dtype = torch.bfloat16

    print(f"Loading transformer from {model_id}...")
    transformer = CustomSD3Transformer2DModel.from_pretrained(
        model_id,
        subfolder="transformer",
        torch_dtype=dtype
    )

    print("Loading pipeline...")
    pipe = CustomStableDiffusion3Pipeline.from_pretrained(
        model_id,
        transformer=transformer,
        torch_dtype=dtype
    ).to("cuda")

    # Define prompt and concepts to track
    prompt = "A cat sitting on grass in a park with trees"
    concepts = ["cat", "grass", "sky", "tree"]

    print(f"Generating image for prompt: {prompt}")
    print(f"Tracking concepts: {concepts}")

    # Generate image with concept attention
    result, concept_attention_outputs = pipe(
        prompt=prompt,
        concepts=concepts,
        num_inference_steps=4,
        guidance_scale=0.0,  # SD3.5 Turbo doesn't need guidance
        height=1024,
        width=1024,
    )

    image = result.images[0]

    # Save the generated image
    image_path = os.path.join(OUTPUT_DIR, "image.png")
    image.save(image_path)
    print(f"Saved generated image to {image_path}")

    # Process concept attention outputs to create heatmaps
    concept_vectors = concept_attention_outputs["concept_output_vectors"]
    image_vectors = concept_attention_outputs["image_output_vectors"]

    # Average across timesteps
    concept_vectors = torch.mean(concept_vectors, dim=0)
    image_vectors = torch.mean(image_vectors, dim=0)

    # Drop the padding token
    concept_vectors = concept_vectors[:, :, 1:]

    # Normalize concept vectors
    concept_vectors = concept_vectors / (concept_vectors.norm(dim=-1, keepdim=True) + 1e-8)

    # Compute concept heatmaps
    concept_heatmaps = einops.einsum(
        concept_vectors.float(),
        image_vectors.float(),
        "layers batch concepts dims, layers batch pixels dims -> batch layers concepts pixels"
    )
    concept_heatmaps = concept_heatmaps[0]  # Remove batch dimension

    # Apply softmax over concept dimension
    concept_heatmaps = torch.nn.functional.softmax(concept_heatmaps, dim=-2)

    # Average across layers
    concept_heatmaps = einops.reduce(
        concept_heatmaps,
        "layers concepts pixels -> concepts pixels",
        reduction="mean"
    )

    # Reshape to spatial dimensions (64x64 for 1024x1024 image with patch_size=2 and vae_scale_factor=8)
    h = w = 64
    concept_heatmaps = einops.rearrange(
        concept_heatmaps,
        "concepts (h w) -> concepts h w",
        h=h,
        w=w
    )
    concept_heatmaps = concept_heatmaps.cpu().float()

    # Save heatmaps for each concept
    for i, concept in enumerate(concepts):
        heatmap = concept_heatmaps[i].numpy()

        fig = plt.figure(figsize=(8, 8), frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(heatmap, cmap='inferno', interpolation='bilinear')

        output_path = os.path.join(OUTPUT_DIR, f"{concept}.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Saved heatmap for '{concept}' to {output_path}")

    print("Done!")
