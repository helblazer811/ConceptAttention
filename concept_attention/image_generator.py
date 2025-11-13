import torch
from PIL import Image
import time
import numpy as np
from einops import rearrange
from transformers import pipeline

from concept_attention.flux.src.flux.cli import SamplingOptions
from concept_attention.flux.src.flux.sampling import denoise, get_noise, get_schedule, prepare, unpack
from concept_attention.flux.src.flux.util import configs, embed_watermark, load_ae, load_clip, load_t5

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as load_sft

from concept_attention.modified_double_stream_block import ModifiedDoubleStreamBlock
from concept_attention.modified_flux_dit import ModifiedFluxDiT
from concept_attention.utils import embed_concepts

def load_flow_model(
    name: str, 
    device: str | torch.device = "cuda", 
    hf_download: bool = True, 
    attention_block_class=ModifiedDoubleStreamBlock,
    dit_class=ModifiedFluxDiT
):
    # Loading Flux
    print("Init model")
    ckpt_path = configs[name].ckpt_path
    if (
        ckpt_path is None
        and configs[name].repo_id is not None
        and configs[name].repo_flow is not None
        and hf_download
    ):
        ckpt_path = hf_hub_download(configs[name].repo_id, configs[name].repo_flow)

    with torch.device("meta" if ckpt_path is not None else device):
        model = dit_class(configs[name].params, attention_block_class=attention_block_class).to(torch.bfloat16)

    if ckpt_path is not None:
        print("Loading checkpoint")
        # load_sft doesn't support torch.device
        sd = load_sft(ckpt_path, device=str(device))
        missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
        # print_load_warning(missing, unexpected)

    return model

def get_models(
    name: str, 
    device: torch.device, 
    offload: bool, 
    is_schnell: bool, 
    attention_block_class=ModifiedDoubleStreamBlock,
    dit_class=ModifiedFluxDiT
):
    t5 = load_t5(device, max_length=256 if is_schnell else 512)
    clip = load_clip(device)
    model = load_flow_model(name, device="cpu" if offload else device, attention_block_class=attention_block_class, dit_class=dit_class)
    ae = load_ae(name, device="cpu" if offload else device)

    return model, ae, t5, clip, None

class FluxGenerator():

    def __init__(
        self, 
        model_name: str, 
        device: str, 
        offload: bool, 
        attention_block_class=ModifiedDoubleStreamBlock,
        dit_class=ModifiedFluxDiT
    ):
        self.device = torch.device(device)
        self.offload = offload
        self.model_name = model_name
        self.is_schnell = model_name == "flux-schnell"
        self.model, self.ae, self.t5, self.clip, self.nsfw_classifier = get_models(
            model_name,
            device=self.device,
            offload=self.offload,
            is_schnell=self.is_schnell,
            attention_block_class=attention_block_class,
            dit_class=dit_class
        )
        
    @torch.inference_mode()
    def generate_image(
        self,
        width,
        height,
        num_steps,
        guidance,
        seed,
        prompt,
        concepts,
        init_image=None,
        image2image_strength=0.0,
        add_sampling_metadata=True,
        restrict_clip_guidance=False,
        joint_attention_kwargs=None,
    ):
        seed = int(seed)
        if seed == -1:
            seed = None

        opts = SamplingOptions(
            prompt=prompt,
            width=width,
            height=height,
            num_steps=num_steps,
            guidance=guidance,
            seed=seed,
        )

        if opts.seed is None:
            opts.seed = torch.Generator(device="cpu").seed()
        print(f"Generating '{opts.prompt}' with seed {opts.seed}")
        t0 = time.perf_counter()

        if init_image is not None:
            if isinstance(init_image, np.ndarray):
                init_image = torch.from_numpy(init_image).permute(2, 0, 1).float() / 255.0
                init_image = init_image.unsqueeze(0) 
            init_image = init_image.to(self.device)
            init_image = torch.nn.functional.interpolate(init_image, (opts.height, opts.width))
            if self.offload:
                self.ae.encoder.to(self.device)
            init_image = self.ae.encode(init_image.to())
            if self.offload:
                self.ae = self.ae.cpu()
                torch.cuda.empty_cache()

        # prepare input
        x = get_noise(
            1,
            opts.height,
            opts.width,
            device=self.device,
            dtype=torch.bfloat16,
            seed=opts.seed,
        )
        timesteps = get_schedule(
            opts.num_steps,
            x.shape[-1] * x.shape[-2] // 4,
            shift=(not self.is_schnell),
        )
        if init_image is not None:
            t_idx = int((1 - image2image_strength) * num_steps)
            t = timesteps[t_idx]
            timesteps = timesteps[t_idx:]
            x = t * x + (1.0 - t) * init_image.to(x.dtype)

        if self.offload:
            self.t5, self.clip = self.t5.to(self.device), self.clip.to(self.device)
        inp = prepare(t5=self.t5, clip=self.clip, img=x, prompt=opts.prompt, restrict_clip_guidance=restrict_clip_guidance)

        ############ Encode the concept ############
        concept_embeddings, concept_ids, concept_vec = embed_concepts(
            self.clip,
            self.t5,
            concepts,
        )
        inp["concepts"] = concept_embeddings.to(x.device)
        inp["concept_ids"] = concept_ids.to(x.device)
        inp["concept_vec"] = concept_vec.to(x.device)
        ###########################################
        # offload TEs to CPU, load model to gpu
        if self.offload:
            self.t5, self.clip = self.t5.cpu(), self.clip.cpu()
            torch.cuda.empty_cache()
            self.model = self.model.to(self.device)
        # denoise initial noise
        x, _, concept_attention_dict = denoise(
            self.model, 
            **inp, 
            timesteps=timesteps, 
            guidance=opts.guidance, 
            joint_attention_kwargs=joint_attention_kwargs
        )
        # offload model, load autoencoder to gpu
        # if self.offload:
        self.model.cpu()
        torch.cuda.empty_cache()
        self.ae.decoder.to(x.device)

        # decode latents to pixel space
        x = unpack(x.float(), opts.height, opts.width)
        # with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
        x = self.ae.decode(x.to(torch.float32).to(self.device))

        self.ae.decoder.cpu()
        torch.cuda.empty_cache()
        self.model.to(self.device)

        t1 = time.perf_counter()

        print(f"Done in {t1 - t0:.1f}s.")
        # bring into PIL format
        x = x.clamp(-1, 1)
        x = embed_watermark(x.float())
        x = rearrange(x[0], "c h w -> h w c")

        img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())

        return img, concept_attention_dict