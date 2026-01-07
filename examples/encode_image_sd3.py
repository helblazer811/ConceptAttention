"""
    Encode an existing image and produce concept heatmaps using Stable Diffusion 3.
"""
import os
import torch
import einops
import matplotlib.pyplot as plt
from PIL import Image

from concept_attention.sd3.pipeline import CustomStableDiffusion3Pipeline, calculate_shift, retrieve_timesteps
from concept_attention.sd3.custom_mmdit import CustomSD3Transformer2DModel
from diffusers.utils.torch_utils import randn_tensor

def retrieve_latents(encoder_output, generator=None, sample_mode="sample"):
    """Extract latents from VAE encoder output."""
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")

if __name__ == "__main__":
    # Create output directory
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "results", "sd3")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load the model
    model_id = "stabilityai/stable-diffusion-3.5-large-turbo"
    dtype = torch.bfloat16
    device = "cuda"

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
    ).to(device)

    # Load the image to encode
    image_path = os.path.join(os.path.dirname(__file__), "..", "images", "dragon_image.png")
    if not os.path.exists(image_path):
        print(f"Image not found at {image_path}, using a test prompt to generate one first")
        # Generate a simple test image
        result, _ = pipe(
            prompt="A fire breathing dragon on a rock",
            concepts=["dragon"],
            num_inference_steps=4,
            guidance_scale=0.0,
            height=1024,
            width=1024,
        )
        image = result.images[0]
    else:
        image = Image.open(image_path).convert("RGB")

    # Define concepts to track
    concepts = ["dragon", "rock", "sky", "fire"]
    caption = "A fire breathing dragon on a rock"
    height, width = 1024, 1024
    num_inference_steps = 4
    timestep_index = -2

    print(f"Encoding image with caption: {caption}")
    print(f"Tracking concepts: {concepts}")

    # Preprocess the image
    image_tensor = pipe.image_processor.preprocess(image, height=height, width=width).to(device=device, dtype=dtype)

    # Encode with VAE
    init_latents = retrieve_latents(pipe.vae.encode(image_tensor))

    # Set up scheduler
    scheduler_kwargs = {}
    if pipe.scheduler.config.get("use_dynamic_shifting", None):
        image_seq_len = (int(height) // pipe.vae_scale_factor // pipe.transformer.config.patch_size) * (
            int(width) // pipe.vae_scale_factor // pipe.transformer.config.patch_size
        )
        mu = calculate_shift(
            image_seq_len,
            pipe.scheduler.config.get("base_image_seq_len", 256),
            pipe.scheduler.config.get("max_image_seq_len", 4096),
            pipe.scheduler.config.get("base_shift", 0.5),
            pipe.scheduler.config.get("max_shift", 1.16),
        )
        scheduler_kwargs["mu"] = mu

    timesteps, _ = retrieve_timesteps(pipe.scheduler, num_inference_steps, device, sigmas=None, **scheduler_kwargs)
    latent_timestep = pipe.scheduler.timesteps[timestep_index * pipe.scheduler.order]
    latent_timestep = latent_timestep.unsqueeze(0)

    # Scale latents
    init_latents = (init_latents - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
    init_latents = torch.cat([init_latents], dim=0)

    # Add noise
    noise = randn_tensor(init_latents.shape, generator=None, device=device, dtype=dtype)
    noisy_latents = pipe.scheduler.scale_noise(init_latents, latent_timestep, noise)
    noisy_latents = noisy_latents.to(device=device, dtype=dtype)

    # Encode the prompt
    prompt_embeds, _, pooled_prompt_embeds, _ = pipe.encode_prompt(caption, None, None)

    # Encode the concepts
    concept_embeds = pipe.encode_concepts(concepts)

    # Get timestep
    timestep = timesteps[timestep_index].expand(noisy_latents.shape[0])

    # Run the transformer
    with torch.no_grad():
        noise_pred, concept_attention_outputs = pipe.transformer(
            hidden_states=noisy_latents,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            concept_hidden_states=concept_embeds,
            pooled_projections=pooled_prompt_embeds,
            joint_attention_kwargs=None,
            return_dict=False,
        )

    # Process concept attention outputs to create heatmaps
    concept_vectors = concept_attention_outputs["concept_output_vectors"]
    image_vectors = concept_attention_outputs["image_output_vectors"]

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

    # Reshape to spatial dimensions
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

        output_path = os.path.join(OUTPUT_DIR, f"encoded_{concept}.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Saved heatmap for '{concept}' to {output_path}")

    print("Done!")
