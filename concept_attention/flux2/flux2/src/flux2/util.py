import base64
import io
import os
import sys

import huggingface_hub
import torch
from PIL import Image
from safetensors.torch import load_file as load_sft

from .autoencoder import AutoEncoder, AutoEncoderParams
from .model import Flux2, Flux2Params
from .text_encoder import Mistral3SmallEmbedder

FLUX2_MODEL_INFO = {
    "flux.2-dev": {
        "repo_id": "black-forest-labs/FLUX.2-dev",
        "filename": "flux2-dev.safetensors",
        "filename_ae": "ae.safetensors",
        "params": Flux2Params(),
    }
}


def load_flow_model(model_name: str, debug_mode: bool = False, device: str | torch.device = "cuda") -> Flux2:
    config = FLUX2_MODEL_INFO[model_name.lower()]

    if debug_mode:
        config["params"].depth = 1
        config["params"].depth_single_blocks = 1
    else:
        if "FLUX2_MODEL_PATH" in os.environ:
            weight_path = os.environ["FLUX2_MODEL_PATH"]
            assert os.path.exists(weight_path), f"Provided weight path {weight_path} does not exist"
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
            model = Flux2(FLUX2_MODEL_INFO[model_name.lower()]["params"]).to(torch.bfloat16)
        print(f"Loading {weight_path} for the FLUX.2 weights")
        sd = load_sft(weight_path, device=str(device))
        model.load_state_dict(sd, strict=False, assign=True)
        return model.to(device)
    else:
        with torch.device(device):
            return Flux2(FLUX2_MODEL_INFO[model_name.lower()]["params"]).to(torch.bfloat16)


def load_mistral_small_embedder(device: str | torch.device = "cuda") -> Mistral3SmallEmbedder:
    return Mistral3SmallEmbedder().to(device)


def load_ae(model_name: str, device: str | torch.device = "cuda") -> AutoEncoder:
    config = FLUX2_MODEL_INFO[model_name.lower()]

    if "AE_MODEL_PATH" in os.environ:
        weight_path = os.environ["AE_MODEL_PATH"]
        assert os.path.exists(weight_path), f"Provided weight path {weight_path} does not exist"
    else:
        # download from huggingface
        try:
            weight_path = huggingface_hub.hf_hub_download(
                repo_id=config["repo_id"],
                filename=config["filename_ae"],
                repo_type="model",
            )
        except huggingface_hub.errors.RepositoryNotFoundError:
            print(
                f"Failed to access the model repository. Please check your internet "
                f"connection and make sure you've access to {config['repo_id']}."
                "Stopping."
            )
            sys.exit(1)

    if isinstance(device, str):
        device = torch.device(device)
    with torch.device("meta"):
        ae = AutoEncoder(AutoEncoderParams())

    print(f"Loading {weight_path} for the AutoEncoder weights")
    sd = load_sft(weight_path, device=str(device))
    ae.load_state_dict(sd, strict=True, assign=True)
    return ae.to(device)


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str
