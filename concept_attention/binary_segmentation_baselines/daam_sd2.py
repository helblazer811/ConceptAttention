import PIL
import torch
from daam import trace
from diffusers import DiffusionPipeline, StableDiffusionPipeline
from diffusers.utils.torch_utils import randn_tensor

import matplotlib.pyplot as plt

from concept_attention.segmentation import SegmentationAbstractClass

def retrieve_latents(encoder_output, generator, sample_mode="sample"):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")

class DAAMStableDiffusion2SegmentationModel(SegmentationAbstractClass):

    def __init__(self, device='cuda:3'):
        # Load the SDXL Pipeline
        model_id = 'stabilityai/stable-diffusion-2-base'
        self.pipeline = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True)
        self.pipeline = self.pipeline.to(device)
        self.device = device

    def _encode_image(self, image: PIL.Image.Image, timestep, height=512, width=512):
        # Preprocess the image
        init_image = self.pipeline.image_processor.preprocess(
            image, 
            height=height, 
            width=width, 
        )
        init_image = init_image.to(dtype=torch.float32) # Make sure float 32 cause otherwise vae encoder doesnt work
        init_image = init_image.to(device=self.device)
        init_latents = retrieve_latents(self.pipeline.vae.encode(init_image), generator=None)
        init_latents = self.pipeline.vae.config.scaling_factor * init_latents
        init_latents = torch.cat([init_latents], dim=0)
        shape = init_latents.shape
        # Add noise
        noise = randn_tensor(shape, generator=None, device=self.device, dtype=self.pipeline.dtype)
        init_latents = self.pipeline.scheduler.add_noise(init_latents, noise, timestep)
        latents = init_latents

        return latents

    @torch.no_grad()
    def _model_forward_pass(
        self, 
        image, 
        prompt,
        timestep=49,
        guidance_scale=1.0,
        num_inference_steps=50,
        height=512,
        width=512,
        dtype=torch.float32,
        batch_size=1,
        generator=None,
    ):
        # Set up timesteps
        self.pipeline.scheduler.set_timesteps(num_inference_steps)
        timestep = self.pipeline.scheduler.timesteps[timestep] # .to(device=device, dtype=dtype)
        # # Encode the image
        # self.pipeline(
        #     image, 
        #     device=self.device,
        #     num_images_per_prompt=1,
        #     output_hidden_states=None,
        # )
        ########################## Prepare latents ##########################
        image_latents = self._encode_image(
            image,
            timestep
        )
        # Add noise at the appropriate timescale
        # noise = randn_tensor(image_latents.shape, generator=generator, device=torch.device(self.device), dtype=dtype)
        # noisy_latents = self.pipeline.scheduler.add_noise(image_latents, noise, timestep.unsqueeze(0))
        # noisy_latents = self.pipeline.scheduler.scale_model_input(noisy_latents, timestep)
        # noisy_latents = noisy_latents.to(device=self.device, dtype=dtype)
        # Encode the prompt
        prompt_embeds, negative_prompt_embeds = self.pipeline.encode_prompt(
            prompt,
            self.device,
            1,
            True,
            None,
            # prompt_embeds=prompt_embeds,
            # negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=0.0,
            # clip_skip=self.pipeline.clip_skip,
        )
        ########################## Run forward pass ##########################
        noise_pred = self.pipeline.unet(
            image_latents,
            timestep,
            encoder_hidden_states=prompt_embeds,
            timestep_cond=None,
            cross_attention_kwargs=None,
            added_cond_kwargs=None,
            return_dict=False,
        )[0]
        ########################## Get and save predicted image ##########################
        # image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[0]
        # do_denormalize = [True] * image.shape[0]
        # image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)
        # # Manually do the logic for the scheduler to get the original prediction
        # s_churn = 0.0
        # s_tmin = 0.0
        # s_tmax = float("inf")
        # s_noise = 1.0
        # # Upcast to avoid precision issues when computing prev_sample
        # sample = noisy_latents.to(torch.float32)
        # sigma = self.pipeline.scheduler.sigmas[self.pipeline.scheduler.index_for_timestep(timestep)]
        # gamma = min(s_churn / (len(self.pipeline.scheduler.sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigma <= s_tmax else 0.0
        # noise = randn_tensor(
        #     noise_pred.shape, dtype=noise_pred.dtype, device=noise_pred.device, generator=generator
        # )
        # eps = noise * s_noise
        # sigma_hat = sigma * (gamma + 1)
        # if gamma > 0:
        #     sample = sample + eps * (sigma_hat**2 - sigma**2) ** 0.5
        # pred_original_sample = sample - sigma_hat * noise_pred
        # # For testing purposes get the predicted original latents and generate the image for it to verify that the image was encoded properly. 
        # image = self.pipeline.vae.decode(pred_original_sample / self.pipeline.vae.config.scaling_factor, return_dict=False, generator=generator)[0]
        # image = self.pipeline.image_processor.postprocess(image, output_type="pil", do_denormalize=[True for _ in range(batch_size)])
        return None

    def segment_individual_image(self, image: torch.Tensor, concepts: list[str], caption: str, **kwargs):
        # Cocnat the concepts into the prompt
        modified_caption = caption + ", ".join([f"a {concept}" for concept in concepts])
        # Run the forward pass with daam trace wrapper
        concept_heatmaps = []
        with trace(self.pipeline) as tc:
            _ = self._model_forward_pass(
                image, 
                caption,
                timestep=49,
                guidance_scale=7.0,
                num_inference_steps=50,
                height=512,
                width=512,
                dtype=torch.float32,
                batch_size=1,
            )

            heat_map = tc.compute_global_heat_map(prompt=modified_caption)
            # For each concept make a heatmap
            for concept in concepts:
                concept_heat_map = heat_map.compute_word_heat_map(concept).heatmap
                concept_heatmaps.append(concept_heat_map)

        concept_heatmaps = torch.stack(concept_heatmaps, dim=0)

        return concept_heatmaps, None
