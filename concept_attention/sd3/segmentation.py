import torch
import einops
import PIL
import matplotlib.pyplot as plt
from typing import Optional

from hopfield_saes.flux_sparse_coding.segmentation import SegmentationAbstractClass
from hopfield_saes.flux_sparse_coding.utils import linear_normalization
from hopfield_saes.flux_sparse_coding.binary_segmentation_baselines.sd3_src.pipeline import CustomStableDiffusion3Pipeline, CustomSD3Transformer2DModel, calculate_shift, retrieve_timesteps
from diffusers.utils.torch_utils import randn_tensor

def load_sd3_turbo_pipeline(device="cuda"):
    transformer = CustomSD3Transformer2DModel.from_pretrained(
        "stabilityai/stable-diffusion-3.5-large-turbo",
        subfolder="transformer",
        torch_dtype=torch.bfloat16
    )
    pipe = CustomStableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-large-turbo", 
        torch_dtype=torch.bfloat16,
        transformer=transformer
    )
    pipe = pipe.to(device)

    return pipe

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


class SD3SegmentationModel(SegmentationAbstractClass):
    
    def __init__(self, mode="concept_attention", device="cuda"):
        self.device = device
        # Set the mode
        assert mode in ["concept_attention", "cross_attention"]
        self.mode = mode
        # Load the pipeline
        self.pipe = load_sd3_turbo_pipeline(device=device)
    
    @torch.no_grad()
    def encode_image(
        self,
        image,
        height=1024,
        width=1024,
        mu=None,
        dtype=None,
        num_inference_steps=4,
        timestep_index=-1
    ):
        # Preprocess the image
        image = self.pipe.image_processor.preprocess(image, height=height, width=width).to(device=self.device, dtype=dtype)
        # Encode with VAE
        init_latents = retrieve_latents(self.pipe.vae.encode(image), generator=None)
        # Convert the int timestep into appropriate level
        scheduler_kwargs = {}
        if self.pipe.scheduler.config.get("use_dynamic_shifting", None) and mu is None:
            image_seq_len = (int(height) // self.pipe.vae_scale_factor // self.pipe.transformer.config.patch_size) * (
                int(width) // self.pipe.vae_scale_factor // self.pipe.transformer.config.patch_size
            )
            mu = calculate_shift(
                image_seq_len,
                self.pipe.scheduler.config.get("base_image_seq_len", 256),
                self.pipe.scheduler.config.get("max_image_seq_len", 4096),
                self.pipe.scheduler.config.get("base_shift", 0.5),
                self.pipe.scheduler.config.get("max_shift", 1.16),
            )
            scheduler_kwargs["mu"] = mu
        elif mu is not None:
            scheduler_kwargs["mu"] = mu
        timesteps, num_inference_steps = retrieve_timesteps(
            self.pipe.scheduler, num_inference_steps, self.device, sigmas=None, **scheduler_kwargs
        )
        # timesteps, num_inference_steps = self.pipe.get_timesteps(num_inference_steps, strength, self.device)
        latent_timestep =  self.pipe.scheduler.timesteps[timestep_index * self.pipe.scheduler.order]
        latent_timestep = latent_timestep.unsqueeze(0)
        # latent_timestep = timesteps[:1]
        # Scale
        init_latents = (init_latents - self.pipe.vae.config.shift_factor) * self.pipe.vae.config.scaling_factor
        init_latents = torch.cat([init_latents], dim=0)
        # Add appropariate noise
        shape = init_latents.shape
        noise = randn_tensor(shape, generator=None, device=self.device, dtype=dtype)
        init_latents = self.pipe.scheduler.scale_noise(init_latents, latent_timestep, noise)
        latents = init_latents.to(device=self.device, dtype=dtype)

        return latents
    
    def decode_image_from_noise_pred(self, noise_pred, timestep, latents):
        # a. Run the scheduler and decode the image
        # compute the previous noisy sample x_t -> x_t-1
        latents_dtype = latents.dtype
        latents = self.pipe.scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]
        # b. Decode the image
        latents = (latents / self.pipe.vae.config.scaling_factor) + self.pipe.vae.config.shift_factor
        image = self.pipe.vae.decode(latents, return_dict=False)[0]
        image = self.pipe.image_processor.postprocess(image, output_type="pil")

        return image
    
    @torch.no_grad()
    def segment_individual_image(
        self, 
        image, 
        concepts, 
        caption, 
        timestep_index=-2,
        layer_range=(0, 38),
        softmax=True,
        num_inference_steps=4,
        # num_samples=1,
        decode_image=False,
        concept_scale_factors=None,
        **kwargs
    ):
        if concept_scale_factors is None:
            concept_scale_factors = [1.0] * len(concepts)
            concept_scale_factors = torch.tensor(concept_scale_factors).to(self.device)
        # Encode the prompt
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = self.pipe.encode_prompt(
            caption,
            None,
            None
        )
        # Encode the image 
        noisy_latents = self.encode_image(
            image, 
            dtype=prompt_embeds.dtype,
            timestep_index=timestep_index,
        )
        # Encode the concepts
        concept_embeds = self.pipe.encode_concepts(concepts)
        # Pull out the correct timestep from the scheduler
        timesteps, num_inference_steps = retrieve_timesteps(
            self.pipe.scheduler, 
            num_inference_steps, 
            self.device, 
            sigmas=None
        )
        timestep = timesteps[timestep_index]
        timestep = timestep.expand(noisy_latents.shape[0])
        # Run the DiT
        noise_pred, concept_attention_outputs = self.pipe.transformer(
            hidden_states=noisy_latents,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            concept_hidden_states=concept_embeds,
            pooled_projections=pooled_prompt_embeds,
            joint_attention_kwargs=None,
            return_dict=False,
        )
        # Actually compute the predicted image to make sure the model is working
        if decode_image:
            image = self.decode_image_from_noise_pred(
                noise_pred=noise_pred,
                timestep=timestep,
                latents=noisy_latents
            )[0]
        else: 
            image = None
        # Unpack the condept vectors and iamge vectors
        concept_vectors = concept_attention_outputs["concept_output_vectors"]
        image_vectors = concept_attention_outputs["image_output_vectors"]
        # Drop the padding token from the prompt embeddings
        concept_vectors = concept_vectors[:, :, 1:]
        # Scale the concept vectors
        print(concept_vectors.shape)
        concept_vectors = concept_vectors * concept_scale_factors.view(1, 1, -1, 1).to(concept_vectors.device)
        # Pull out layers of interest
        concept_vectors = concept_vectors[layer_range[0]:layer_range[1]]
        image_vectors = image_vectors[layer_range[0]:layer_range[1]]
        # Pull out only the first N concepts
        # concept_vectors = concept_vectors[:, :, :, :3]
        concept_vectors = linear_normalization(concept_vectors, dim=-2)
        # Now compute the heatmap
        concept_heatmaps = einops.einsum(
            concept_vectors.float(),
            image_vectors.float(),
            "layers batch concepts dims, layers batch pixels dims -> batch layers concepts pixels"
        )
        concept_heatmaps = concept_heatmaps[0]
        if softmax:
            concept_heatmaps = torch.nn.functional.softmax(concept_heatmaps, dim=-2)

        concept_heatmaps = einops.reduce(
            concept_heatmaps,
            "layers concepts pixels -> concepts pixels",
            reduction="mean"
        )
        concept_heatmaps = einops.rearrange(
            concept_heatmaps,
            "concepts (h w) -> concepts h w",
            h=64,
            w=64
        )
        concept_heatmaps = concept_heatmaps.cpu().float()

        if self.mode == "cross_attention":
            # Unpack the cross attention maps
            cross_attention_maps = concept_attention_outputs["cross_attention_maps"]
            print(f"Cross attention maps shape: {cross_attention_maps.shape}")
            # Apply softmax over the concept dimension
            cross_attention_maps = torch.nn.functional.softmax(cross_attention_maps, dim=-2)
            # Pull out the concept of interest which is index 2 
            cross_attention_maps = cross_attention_maps[:, :, 2]
            # Drop the batch dimension
            cross_attention_maps = cross_attention_maps[:, 0]
            # Pull out the layers of interest
            cross_attention_maps = cross_attention_maps[layer_range[0]:layer_range[1]]
            # # Drop the padding token from the cross attention maps
            # cross_attention_maps = cross_attention_maps[:, :, 1:]
            # # Pull out the ones corresponding to concepts of interest
            # cross_attention_maps = cross_attention_maps[:, :, :len(concepts)]
            # # Drop the batch dimension
            # cross_attention_maps = cross_attention_maps[:, 0]
            # # Pull out the layers of interest
            # cross_attention_maps = cross_attention_maps[layer_range[0]:layer_range[1]]
            # # Perform softmax
            # cross_attention_maps = torch.nn.functional.softmax(cross_attention_maps, dim=-2)
            # Average across layers and time 
            cross_attention_maps = einops.reduce(
                cross_attention_maps,
                "layers pixels -> pixels",
                reduction="mean"
            )
            # Repeat for len(concepts) times
            cross_attention_maps = cross_attention_maps.repeat(len(concepts), 1)
            # Reshape into the correct format
            cross_attention_maps = einops.rearrange(
                cross_attention_maps,
                "concepts (h w) -> concepts h w",
                h=64,
                w=64
            )
            cross_attention_maps = cross_attention_maps.cpu().float()

            return cross_attention_maps, image

        return concept_heatmaps, image