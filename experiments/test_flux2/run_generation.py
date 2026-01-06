"""
Minimal script to load FLUX2 model and generate images.
Stripped down from cli.py to just the essentials.
"""

import os
import random
import sys
from pathlib import Path


import huggingface_hub
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation, PillowWriter
import torch
from torch import Tensor
from einops import rearrange
from PIL import Image
from safetensors.torch import load_file as load_sft
from tqdm import tqdm

from concept_attention.flux2.flux2.src.flux2.model import Flux2, Flux2Params
from concept_attention.flux2.flux2.src.flux2.sampling import (
    batched_prc_img,
    batched_prc_txt,
    encode_image_refs,
    get_schedule,
    scatter_ids,
)
from concept_attention.flux2.flux2.src.flux2.util import load_ae, load_mistral_small_embedder

from concept_attention.flux2.modified_dit import ModifiedFlux2

import einops

# Model configuration
FLUX2_MODEL_INFO = {
    "flux.2-dev": {
        "repo_id": "black-forest-labs/FLUX.2-dev",
        "filename": "flux2-dev.safetensors",
        "filename_ae": "ae.safetensors",
        "params": Flux2Params(),
    }
}


def denoise(
    model: Flux2,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    concepts: Tensor,
    concept_ids: Tensor,
    # sampling parameters
    timesteps: list[float],
    guidance: float,
    # extra img tokens (sequence-wise)
    img_cond_seq: Tensor | None = None,
    img_cond_seq_ids: Tensor | None = None,
):
    guidance_vec = torch.full(
        (img.shape[0],), guidance, device=img.device, dtype=img.dtype
    )
    concept_attention_dicts = []
    noisy_images = []

    for t_curr, t_prev in tqdm(
        zip(timesteps[:-1], timesteps[1:]), total=len(timesteps) - 1, desc="Denoising"
    ):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        img_input = img
        img_input_ids = img_ids
        if img_cond_seq is not None:
            assert (
                img_cond_seq_ids is not None
            ), "You need to provide either both or neither of the sequence conditioning"
            img_input = torch.cat((img_input, img_cond_seq), dim=1)
            img_input_ids = torch.cat((img_input_ids, img_cond_seq_ids), dim=1)

        # Store the noisy img_input for animation
        noisy_images.append(
            (img_input[:, : img.shape[1]], img_input_ids[:, : img.shape[1]])
        )

        pred, current_concept_attention_dict = model(
            x=img_input,
            x_ids=img_input_ids,
            timesteps=t_vec,
            ctx=txt,
            ctx_ids=txt_ids,
            guidance=guidance_vec,
            concepts=concepts,
            concept_ids=concept_ids,
        )
        concept_attention_dicts.append(current_concept_attention_dict)
        if img_input_ids is not None:
            pred = pred[:, : img.shape[1]]

        img = img + (t_prev - t_curr) * pred

    return img, concept_attention_dicts, noisy_images


def find_first_different_token_index(mistral, token1: str, token2: str) -> int:
    """
    Compare embeddings of two tokens and find the index where they first differ.

    Args:
        mistral: The Mistral text embedder
        token1: First token/word (e.g., "dog")
        token2: Second token/word (e.g., "cat")

    Returns:
        Index of the first position where embeddings differ
    """
    # Encode both tokens
    with torch.no_grad():
        emb1 = mistral([token1]).to(torch.bfloat16)
        emb2 = mistral([token2]).to(torch.bfloat16)

    print(f"Token '{token1}' embedding shape: {emb1.shape}")
    print(f"Token '{token2}' embedding shape: {emb2.shape}")

    # Find first position where embeddings differ
    # Compare along sequence dimension (axis 1)
    seq_len = min(emb1.shape[1], emb2.shape[1])

    for i in range(seq_len):
        # Check if embeddings at position i are different
        if not torch.allclose(emb1[0, i], emb2[0, i], rtol=1e-5, atol=1e-8):
            print(f"First difference at token index {i}")
            print(f"  '{token1}' embedding norm at {i}: {emb1[0, i].norm().item():.4f}")
            print(f"  '{token2}' embedding norm at {i}: {emb2[0, i].norm().item():.4f}")
            return i

    print("Embeddings are identical across all positions!")
    return -1


def load_flow_model(
    model_name: str, debug_mode: bool = False, device: str | torch.device = "cuda"
) -> Flux2:
    """Load the FLUX2 flow model."""
    config = FLUX2_MODEL_INFO[model_name.lower()]

    if debug_mode:
        config["params"].depth = 1
        config["params"].depth_single_blocks = 1
    else:
        if "FLUX2_MODEL_PATH" in os.environ:
            weight_path = os.environ["FLUX2_MODEL_PATH"]
            assert os.path.exists(
                weight_path
            ), f"Provided weight path {weight_path} does not exist"
        else:
            # download from huggingface
            try:
                weight_path = huggingface_hub.hf_hub_download(
                    repo_id=config["repo_id"],
                    filename=config["filename"],
                    repo_type="model",
                )
            except huggingface_hub.errors.RepositoryNotFoundError:
                print(
                    f"Failed to access the model repository. Please check your internet "
                    f"connection and make sure you've access to {config['repo_id']}."
                    "Stopping."
                )
                sys.exit(1)

    if not debug_mode:
        with torch.device("meta"):
            model = ModifiedFlux2(FLUX2_MODEL_INFO[model_name.lower()]["params"]).to(
                torch.bfloat16
            )
        print(f"Loading {weight_path} for the FLUX.2 weights")
        sd = load_sft(weight_path, device=str(device))
        model.load_state_dict(sd, strict=False, assign=True)
        return model.to(device)
    else:
        with torch.device(device):
            return ModifiedFlux2(FLUX2_MODEL_INFO[model_name.lower()]["params"]).to(
                torch.bfloat16
            )


def load_models(model_name: str = "flux.2-dev", cpu_offloading: bool = False):
    """
    Load FLUX2 models.

    Args:
        model_name: Model name (e.g., "flux.2-dev")
        cpu_offloading: Whether to initially load flow model on CPU

    Returns:
        Tuple of (mistral_embedder, flow_model, autoencoder)
    """
    torch_device = torch.device("cuda")

    print("Loading models...")
    mistral = load_mistral_small_embedder()
    model = load_flow_model(
        model_name, debug_mode=False, device="cpu" if cpu_offloading else torch_device
    )
    ae = load_ae(model_name)
    ae.eval()
    mistral.eval()

    return mistral, model, ae


def generate_image(
    mistral,
    model,
    ae,
    prompt: str,
    concepts: list[str],
    width: int = 1360,
    height: int = 768,
    num_steps: int = 50,
    guidance: float = 4.0,
    seed: int | None = None,
    input_images: list[str] = None,
    output_path: str = "output.png",
    cpu_offloading: bool = False,
):
    """
    Generate an image using FLUX2.

    Args:
        mistral: Loaded Mistral text embedder
        model: Loaded FLUX2 flow model
        ae: Loaded autoencoder
        prompt: Text prompt for generation
        concepts: List of concept names to track attention for
        width: Output width in pixels
        height: Output height in pixels
        num_steps: Number of denoising steps
        guidance: Guidance scale
        seed: Random seed (None for random)
        input_images: List of input image paths for conditioning
        output_path: Path to save output image
        cpu_offloading: Whether to offload models between CPU/GPU

    Returns:
        Tuple of (generated_image, concept_attention_dicts, decoded_noisy_images)
        - generated_image: PIL Image object
        - concept_attention_dicts: List of attention dicts per timestep
        - decoded_noisy_images: List of decoded noisy images for each denoising step
    """
    concept_names = concepts
    torch_device = torch.device("cuda")

    # Set seed
    if seed is None:
        seed = random.randrange(2**31)
    print(f"Using seed: {seed}")

    # Compare token embeddings to find where they differ
    print("\n=== Token Embedding Comparison ===")
    find_first_different_token_index(mistral, "dog", "cat")
    print("==================================\n")

    with torch.no_grad():
        # Load and encode input images if provided
        img_ctx = []
        if input_images:
            img_ctx = [Image.open(img_path) for img_path in input_images]
        ref_tokens, ref_ids = encode_image_refs(ae, img_ctx)

        # Encode text prompt
        print(f"Generating with prompt: {prompt}")
        ctx = mistral([prompt]).to(torch.bfloat16)
        ctx, ctx_ids = batched_prc_txt(ctx)

        # Encode the concepts and pull out their tokens
        concept_embeddings = mistral(concepts).to(torch.bfloat16)
        concepts, concept_ids = batched_prc_txt(concept_embeddings)
        print(f"Concepts shape: {concepts.shape}")
        print(f"Concept IDs shape: {concept_ids.shape}")
        assert concepts.shape[0] == len(
            concepts
        ), "Mismatch in number of concepts and encoded concept tensors"
        concepts = concepts[:, 510].unsqueeze(0)
        concept_ids = concept_ids[:, 510].unsqueeze(0)

        if cpu_offloading:
            mistral = mistral.cpu()
            torch.cuda.empty_cache()
            model = model.to(torch_device)

        # Create noise
        shape = (1, 128, height // 16, width // 16)
        generator = torch.Generator(device="cuda").manual_seed(seed)
        randn = torch.randn(
            shape, generator=generator, dtype=torch.bfloat16, device="cuda"
        )
        x, x_ids = batched_prc_img(randn)

        # Denoise
        timesteps = get_schedule(num_steps, x.shape[1])
        x, concept_attention_dicts, noisy_images = denoise(
            model,
            x,
            x_ids,
            ctx,
            ctx_ids,
            concepts=concepts,
            concept_ids=concept_ids,
            timesteps=timesteps,
            guidance=guidance,
            img_cond_seq=ref_tokens,
            img_cond_seq_ids=ref_ids,
        )

        # Decode the final image
        x = torch.cat(scatter_ids(x, x_ids)).squeeze(2)
        print(f"Latent shape before decoding: {x.shape}")
        x = ae.decode(x).float()

        # Decode all noisy intermediate images
        print("Decoding noisy intermediate images...")
        decoded_noisy_images = []
        for noisy_img, noisy_img_ids in tqdm(
            noisy_images, desc="Decoding noisy images"
        ):
            noisy_decoded = torch.cat(scatter_ids(noisy_img, noisy_img_ids)).squeeze(2)
            noisy_decoded = ae.decode(noisy_decoded).float()
            decoded_noisy_images.append(noisy_decoded)

        if cpu_offloading:
            model = model.cpu()
            torch.cuda.empty_cache()
            mistral = mistral.to(torch_device)

    # Convert to image
    x = x.clamp(-1, 1)
    x = rearrange(x[0], "c h w -> h w c")
    img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())

    # Save image
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, quality=95, subsampling=0)
    print(f"Saved to {output_path}")

    return img, concept_attention_dicts, decoded_noisy_images


def save_concept_heatmaps(
    concept_attention_dicts,
    concept_names: list[str],
    width: int,
    height: int,
    num_steps: int,
    time_indices: list[int] | None = None,
    block_indices: list[int] | None = None,
    apply_softmax: bool = True,
    softmax_temperature: float = 1.0,
    output_dir: str = "output/concept_attention",
):
    """
    Process and save concept attention heatmaps.

    Args:
        concept_attention_dicts: List of attention dicts per timestep from generate_image
        concept_names: List of concept names corresponding to the attention maps
        width: Image width in pixels
        height: Image height in pixels
        num_steps: Number of denoising steps that were used
        time_indices: Specific timestep indices to average over (None = all steps)
        block_indices: Specific block indices to average over (None = all blocks)
        apply_softmax: Whether to apply softmax normalization
        softmax_temperature: Temperature for softmax (higher = softer)
        output_dir: Directory to save heatmap images
    """
    # Set defaults
    if time_indices is None:
        time_indices = list(range(num_steps))
    if block_indices is None:
        num_blocks = len(concept_attention_dicts[0])
        block_indices = list(range(num_blocks))

    # Stack the concept attention dicts over time
    selected_concept_attention = []
    for t_idx in time_indices:
        time_step_dicts = concept_attention_dicts[t_idx]
        selected_blocks = []
        for b_idx in block_indices:
            selected_blocks.append(time_step_dicts[b_idx]["concept_scores"])
        selected_blocks = torch.stack(selected_blocks)
        selected_concept_attention.append(selected_blocks)
    selected_concept_attention = torch.stack(selected_concept_attention)

    # Average the heatmaps
    avg_concept_scores = einops.reduce(
        selected_concept_attention,
        "time blocks batch num_concepts num_image_tokens -> batch num_concepts num_image_tokens",
        "mean",
    )
    # Pull out batch index
    avg_concept_scores = avg_concept_scores[0]

    # Reshape spatially (16 pixels is 1 latent token)
    num_image_tokens_h = height // 16
    num_image_tokens_w = width // 16
    avg_concept_scores = einops.rearrange(
        avg_concept_scores,
        "num_concepts (h w) -> num_concepts h w",
        h=num_image_tokens_h,
        w=num_image_tokens_w,
    )

    # Optionally apply softmax over spatial dimensions with temperature
    if apply_softmax:
        print(f"Applying softmax with temperature={softmax_temperature}")
        avg_concept_scores = torch.softmax(
            avg_concept_scores / softmax_temperature, dim=0
        )

    # Save heatmaps each as images with PIL in the output dir
    os.makedirs(output_dir, exist_ok=True)
    inferno = cm.get_cmap("inferno")

    # Compute global min/max across all concepts for consistent normalization
    global_min = avg_concept_scores.min()
    global_max = avg_concept_scores.max()
    print(f"Global heatmap range: [{global_min:.4f}, {global_max:.4f}]")

    for concept_idx, concept in enumerate(concept_names):
        concept_score_map = avg_concept_scores[concept_idx]

        # Normalize to [0, 1] using global min/max
        concept_score_map = (concept_score_map - global_min) / (global_max - global_min)
        concept_score_map_np = concept_score_map.cpu().float().numpy()

        # Apply inferno colormap
        colored = inferno(concept_score_map_np)
        # Convert to uint8 RGB (discard alpha channel)
        colored_uint8 = (colored[:, :, :3] * 255).astype("uint8")

        concept_img = Image.fromarray(colored_uint8)
        concept_img = concept_img.resize((width, height), resample=Image.NEAREST)
        heatmap_path = os.path.join(output_dir, f"concept_{concept}_heatmap.png")
        concept_img.save(heatmap_path)
        print(f"Saved heatmap for '{concept}' to {heatmap_path}")


def animate_layers_over_time(
    concept_attention_dicts,
    concept_names: list[str],
    layer_indices: list[int],
    width: int,
    height: int,
    apply_softmax: bool = True,
    softmax_temperature: float = 1.0,
    output_dir: str = "output/concept_attention_animations",
    fps: int = 10,
):
    """
    Create animations showing how concept attention evolves over timesteps, averaged over layers.

    Args:
        concept_attention_dicts: List of attention dicts per timestep from generate_image
        concept_names: List of concept names corresponding to the attention maps
        layer_indices: List of layer/block indices to average over
        width: Image width in pixels
        height: Image height in pixels
        apply_softmax: Whether to apply softmax normalization across concepts
        softmax_temperature: Temperature for softmax (higher = softer)
        output_dir: Directory to save animation files
        fps: Frames per second for the animation
    """
    os.makedirs(output_dir, exist_ok=True)

    num_timesteps = len(concept_attention_dicts)
    num_concepts = len(concept_names)

    # Spatial dimensions
    num_image_tokens_h = height // 16
    num_image_tokens_w = width // 16

    # Pre-process all frames for all concepts
    # Shape: [num_timesteps, num_layers, num_concepts, h, w]
    all_frames = []

    for t_idx in range(num_timesteps):
        time_step_dicts = concept_attention_dicts[t_idx]
        layer_frames = []

        for layer_idx in layer_indices:
            # Get concept scores for this layer: [batch, num_concepts, num_image_tokens]
            concept_scores = time_step_dicts[layer_idx]["concept_scores"]
            concept_scores = concept_scores[0]  # Remove batch dimension

            # Reshape spatially
            concept_scores = einops.rearrange(
                concept_scores,
                "num_concepts (h w) -> num_concepts h w",
                h=num_image_tokens_h,
                w=num_image_tokens_w,
            )

            layer_frames.append(concept_scores)

        layer_frames = torch.stack(layer_frames)  # [num_layers, num_concepts, h, w]
        all_frames.append(layer_frames)

    all_frames = torch.stack(
        all_frames
    )  # [num_timesteps, num_layers, num_concepts, h, w]

    # Average over layers
    all_frames = einops.reduce(
        all_frames,
        "timesteps layers concepts h w -> timesteps concepts h w",
        "mean",
    )

    # Optionally apply softmax
    if apply_softmax:
        print(f"Applying softmax with temperature={softmax_temperature}")
        all_frames = torch.softmax(all_frames / softmax_temperature, dim=1)

    # Get global min/max for consistent normalization
    global_min = all_frames.min()
    global_max = all_frames.max()
    print(f"Global animation heatmap range: [{global_min:.4f}, {global_max:.4f}]")

    # Create animation for each concept
    for concept_idx, concept_name in enumerate(concept_names):
        # Create figure with no margins or decorations
        fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
        ax = plt.axes([0, 0, 1, 1])
        ax.axis("off")

        # Initialize with first frame
        frame_data = all_frames[0, concept_idx]
        frame_data = (frame_data - global_min) / (global_max - global_min)
        frame_data_np = frame_data.cpu().float().numpy()

        im = ax.imshow(
            frame_data_np, cmap="inferno", vmin=0, vmax=1, interpolation="nearest"
        )

        # Animation update function
        def update(frame_idx):
            frame_data = all_frames[frame_idx, concept_idx]
            frame_data = (frame_data - global_min) / (global_max - global_min)
            frame_data_np = frame_data.cpu().float().numpy()
            im.set_array(frame_data_np)
            return [im]

        # Create animation
        anim = FuncAnimation(
            fig, update, frames=num_timesteps, interval=1000 // fps, blit=True
        )

        # Save animation
        output_path = os.path.join(
            output_dir, f"concept_{concept_name}_layers_animation.gif"
        )
        writer = PillowWriter(fps=fps)
        anim.save(output_path, writer=writer)
        print(f"Saved animation for '{concept_name}' to {output_path}")

        plt.close(fig)


def animate_denoising_process(
    decoded_noisy_images,
    width: int,
    height: int,
    output_dir: str = "output/denoising_animations",
    output_filename: str = "denoising_process.gif",
    fps: int = 10,
):
    """
    Create animation showing the denoising process over time.

    Args:
        decoded_noisy_images: List of decoded image tensors for each denoising step
        width: Image width in pixels
        height: Image height in pixels
        output_dir: Directory to save animation file
        output_filename: Name of output GIF file
        fps: Frames per second for the animation
    """
    os.makedirs(output_dir, exist_ok=True)

    num_frames = len(decoded_noisy_images)
    print(f"Creating denoising animation with {num_frames} frames...")

    # Create figure
    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
    ax = plt.axes([0, 0, 1, 1])
    ax.axis("off")

    # Convert first frame to displayable format
    frame_data = decoded_noisy_images[0].clamp(-1, 1)
    frame_data = rearrange(frame_data[0], "c h w -> h w c")
    frame_data_np = (127.5 * (frame_data + 1.0)).cpu().byte().numpy()

    im = ax.imshow(frame_data_np, interpolation="nearest")

    # Animation update function
    def update(frame_idx):
        frame_data = decoded_noisy_images[frame_idx].clamp(-1, 1)
        frame_data = rearrange(frame_data[0], "c h w -> h w c")
        frame_data_np = (127.5 * (frame_data + 1.0)).cpu().byte().numpy()
        im.set_array(frame_data_np)
        return [im]

    # Create animation
    anim = FuncAnimation(
        fig, update, frames=num_frames, interval=1000 // fps, blit=True
    )

    # Save animation
    output_path = os.path.join(output_dir, output_filename)
    writer = PillowWriter(fps=fps)
    anim.save(output_path, writer=writer)
    print(f"Saved denoising animation to {output_path}")

    plt.close(fig)


if __name__ == "__main__":
    # Example usage: Load models once, generate multiple images
    mistral, model, ae = load_models(model_name="flux.2-dev")

    # Define parameters
    # concepts = ["dog", "tree", "sun", "sky", "grass", "moon", "cloud", "flower"]
    concepts = ["sky", "sword", "man", "tree", "grass"]
    width = 2048
    height = 2048
    num_steps = 50

    # Generate image
    img, concept_attention_dicts, decoded_noisy_images = generate_image(
        mistral,
        model,
        ae,
        # prompt="A photo of a dog by a tree in a field. Sun in the sky. ",
        prompt="A man with a sword in a field by a tree. ",
        concepts=concepts,
        width=width,
        height=height,
        num_steps=num_steps,
        guidance=4.0,
        seed=44,
        output_path="experiments/test_flux2/output/sample.png",
    )

    # Save concept attention heatmaps
    save_concept_heatmaps(
        concept_attention_dicts=concept_attention_dicts,
        concept_names=concepts,
        width=width,
        height=height,
        num_steps=num_steps,
        time_indices=[37, 38, 39, 40, 41, 42, 43, 44, 45],
        block_indices=[5, 6, 7],
        apply_softmax=True,
        softmax_temperature=1000.0,  # Higher values (e.g., 5.0) make softmax softer/smoother
        output_dir="experiments/test_flux2/output/concept_attention",
    )

    # Create animations showing how attention evolves over time for the last 3 layers
    animate_layers_over_time(
        concept_attention_dicts=concept_attention_dicts,
        concept_names=concepts,
        layer_indices=[5, 6, 7],
        width=width,
        height=height,
        apply_softmax=True,
        softmax_temperature=1000.0,
        output_dir="experiments/test_flux2/output/concept_attention_animations",
        fps=5,
    )

    # Create animation showing the denoising process
    animate_denoising_process(
        decoded_noisy_images=decoded_noisy_images,
        width=width,
        height=height,
        output_dir="experiments/test_flux2/output/denoising_animations",
        output_filename="denoising_process.gif",
        fps=5,
    )
