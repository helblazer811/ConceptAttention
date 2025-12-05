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
import torch
from torch import Tensor
from einops import rearrange
from PIL import Image
from safetensors.torch import load_file as load_sft

from flux2.model import Flux2, Flux2Params
from flux2.sampling import (
    batched_prc_img,
    batched_prc_txt,
    encode_image_refs,
    get_schedule,
    scatter_ids,
)
from flux2.util import load_ae, load_mistral_small_embedder

from modified_dit import ModifiedFlux2

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
    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        img_input = img
        img_input_ids = img_ids
        if img_cond_seq is not None:
            assert (
                img_cond_seq_ids is not None
            ), "You need to provide either both or neither of the sequence conditioning"
            img_input = torch.cat((img_input, img_cond_seq), dim=1)
            img_input_ids = torch.cat((img_input_ids, img_cond_seq_ids), dim=1)

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

    return img, concept_attention_dicts


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
    concept_attention_kwargs=None,
):
    """
    Generate an image using FLUX2.

    Args:
        mistral: Loaded Mistral text embedder
        model: Loaded FLUX2 flow model
        ae: Loaded autoencoder
        prompt: Text prompt for generation
        width: Output width in pixels
        height: Output height in pixels
        num_steps: Number of denoising steps
        guidance: Guidance scale
        seed: Random seed (None for random)
        input_images: List of input image paths for conditioning
        output_path: Path to save output image
        cpu_offloading: Whether to offload models between CPU/GPU
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
        x, concept_attention_dicts = denoise(
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

        # Decode
        x = torch.cat(scatter_ids(x, x_ids)).squeeze(2)
        print(f"Latent shape before decoding: {x.shape}")
        x = ae.decode(x).float()

        if cpu_offloading:
            model = model.cpu()
            torch.cuda.empty_cache()
            mistral = mistral.to(torch_device)

    # Convert to image
    x = x.clamp(-1, 1)
    x = rearrange(x[0], "c h w -> h w c")
    img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, quality=95, subsampling=0)
    print(f"Saved to {output_path}")

    ################## Use the concept attention dict for analysis ##################
    # Unpack concept attention kwargs
    time_indices = concept_attention_kwargs.get("time_indices", None)
    if time_indices is None:
        time_indices = list(range(num_steps))
    block_indices = concept_attention_kwargs.get("block_indices", None)
    if block_indices is None:
        num_blocks = len(concept_attention_dicts[0])
        block_indices = list(range(num_blocks))
    apply_softmax = concept_attention_kwargs.get("apply_softmax", True)
    softmax_temperature = concept_attention_kwargs.get("softmax_temperature", 1.0)
    output_dir = concept_attention_kwargs.get("output_dir", "output/concept_attention")
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
        concept_img.save(
            os.path.join(output_dir, f"concept_{concept}_heatmap.png"),
        )
    # Shape should be list[list[dict]], outer list over timesteps, inner list over blocks
    # Did projection inside to avoid OOM
    # In each dict is {"concept_scores"} with shape (batch_size, num_concepts, num_image_tokens)


if __name__ == "__main__":
    # Example usage: Load models once, generate multiple images
    mistral, model, ae = load_models(model_name="flux.2-dev")

    # Generate first image
    generate_image(
        mistral,
        model,
        ae,
        prompt="A photo of a dog by a tree in a field. Sun in the sky. ",
        concepts=["dog", "tree", "sun", "sky", "grass"],
        # width=1360,
        # height=768,
        width=2048,
        height=2048,
        num_steps=50,
        guidance=4.0,
        seed=42,
        output_path="output/sample.png",
        concept_attention_kwargs={
            "time_indices": [40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
            "block_indices": [5, 6, 7],
            "apply_softmax": True,
            "softmax_temperature": 10.0,  # Higher values (e.g., 5.0) make softmax softer/smoother
            "output_dir": "output/concept_attention",
        },
    )

    # Can generate more images without reloading models
    # generate_image(
    #     mistral,
    #     model,
    #     ae,
    #     prompt="a cat in a hat",
    #     seed=123,
    #     output_path="output/sample2.png",
    # )
