import math
from typing import Callable

from tqdm import tqdm
import torch
from einops import rearrange, repeat
from torch import Tensor

from .model import Flux
from .modules.conditioner import HFEmbedder

def get_noise(
    num_samples: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
):
    return torch.randn(
        num_samples,
        16,
        # allow for packing
        2 * math.ceil(height / 16),
        2 * math.ceil(width / 16),
        device=device,
        dtype=dtype,
        generator=torch.Generator(device=device).manual_seed(seed),
    )

def prepare(t5: HFEmbedder, clip: HFEmbedder, img: Tensor, prompt: str | list[str], restrict_clip_guidance=False) -> dict[str, Tensor]:
    bs, c, h, w = img.shape
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)

    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img.shape[0] == 1 and bs > 1:
        img = repeat(img, "1 ... -> bs ...", bs=bs)

    img_ids = torch.zeros(h // 2, w // 2, 3)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

    if isinstance(prompt, str):
        prompt = [prompt]
    txt = t5(prompt)
    if txt.shape[0] == 1 and bs > 1:
        txt = repeat(txt, "1 ... -> bs ...", bs=bs)
    txt_ids = torch.zeros(bs, txt.shape[1], 3)

    if restrict_clip_guidance:
        vec = clip("")
    else:
        vec = clip(prompt)
    if vec.shape[0] == 1 and bs > 1:
        vec = repeat(vec, "1 ... -> bs ...", bs=bs)

    return {
        "img": img,
        "img_ids": img_ids.to(img.device),
        "txt": txt.to(img.device),
        "txt_ids": txt_ids.to(img.device),
        "vec": vec.to(img.device),
    }

def time_shift(mu: float, sigma: float, t: Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # estimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()

def denoise(
    model: Flux,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    # sampling parameters
    timesteps: list[float],
    guidance: float = 4.0,
    concepts: Tensor = None,
    concept_ids: Tensor = None,
    concept_vec: Tensor = None,
    null_txt_ids: Tensor = None,
    null_txt_vec: Tensor = None,
    null_txt: Tensor = None,
    edit_txt_ids: Tensor = None,
    edit_txt_vec: Tensor = None,
    edit_txt: Tensor = None,
    return_intermediate_images=True,
    edit_metadata=None,
    joint_attention_kwargs=None,
):
    intermediate_images = [img]
    score_deltas = []
    # this is ignored for schnell
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    iteration = 0
    for t_curr, t_prev in tqdm(zip(timesteps[:-1], timesteps[1:])):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        pred = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            concepts=concepts,
            concept_ids=concept_ids,
            concept_vec=concept_vec,
            null_txt=null_txt,
            null_txt_ids=null_txt_ids,
            null_txt_vec=null_txt_vec,
            edit_txt=edit_txt,
            edit_txt_ids=edit_txt_ids,
            edit_txt_vec=edit_txt_vec,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
            edit_metadata=edit_metadata,
            iteration=iteration,
            joint_attention_kwargs=joint_attention_kwargs
        )

        # Save the image score delta
        score_delta = (t_prev - t_curr) * pred
        score_delta = torch.norm(score_delta, p=2, dim=-1)
        score_deltas.append(score_delta)

        img = img + (t_prev - t_curr) * pred
        intermediate_images.append(img)
        # increment iteration
        iteration += 1

    return img, intermediate_images, score_deltas

def unpack(x: Tensor, height: int, width: int) -> Tensor:
    return rearrange(
        x,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=math.ceil(height / 16),
        w=math.ceil(width / 16),
        ph=2,
        pw=2,
    )
