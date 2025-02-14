import PIL
import torch
from daam import trace
from diffusers import DiffusionPipeline
from diffusers.utils.torch_utils import randn_tensor

from concept_attention.segmentation import SegmentationAbstractClass


class DAAMStableDiffusionXLSegmentationModel(SegmentationAbstractClass):

    def __init__(self, device='cuda:3'):
        # Load the SDXL Pipeline
        model_id = 'stabilityai/stable-diffusion-xl-base-1.0'
        self.pipeline = DiffusionPipeline.from_pretrained(
            model_id, 
            use_auth_token=True,
            torch_dtype=torch.float32,
            use_safetensors=True
        )
        self.pipeline = self.pipeline.to(device)
        self.device = device

    def _encode_prompt(self, prompt, guidance_scale=0.0, device="cuda:0"):
        # Get the prompt embeddings
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = self.pipeline.encode_prompt(
            prompt,
            None,
            device,
            True,
            negative_prompt=None,
            # lora_scale=None,
            # clip_skip=None,
        )

        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    def _encode_image(self, image: PIL.Image.Image, generator=None):
        image_latents = self.pipeline.vae.encode(image)
        image_latents = image_latents.latent_dist.sample(generator)
        image_latents = self.pipeline.vae.config.scaling_factor * image_latents

        return image_latents

    def _process_added_kwargs(
        self, 
        prompt_embeds, 
        pooled_prompt_embeds,
        height=512,
        width=512, 
    ):
        add_text_embeds = pooled_prompt_embeds
        if self.pipeline.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.pipeline.text_encoder_2.config.projection_dim
        add_time_ids = self.pipeline._get_add_time_ids(
            (height, width),
            (0, 0),
            (height, width),
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        # Proprocess the text embeddings
        added_cond_kwargs = {
            "time_ids": add_time_ids.to(device=self.device),
            "text_embeds": pooled_prompt_embeds.to(device=self.device),
        }

        return added_cond_kwargs

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
        ########################## Prepare latents ##########################
        init_image = self.pipeline.image_processor.preprocess(
            image, 
            height=height, 
            width=width, 
            # crops_coords=None, 
            # resize_mode="default"
        )
        init_image = init_image.to(dtype=torch.float32) # Make sure float 32 cause otherwise vae encoder doesnt work
        init_image = init_image.to(device=self.device)
        initial_image_latents = self._encode_image(init_image)
        # Figure out the number fo steps to do
        timestep = self.pipeline.scheduler.timesteps[timestep]
        # Encode the prompt
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = self._encode_prompt(
            prompt, 
            guidance_scale=guidance_scale, 
            device=self.device
        )
        # Proprocess the text embeddings
        added_cond_kwargs = self._process_added_kwargs(
            prompt_embeds, 
            pooled_prompt_embeds,
            width=width,
            height=height
        )
        # Add noise at the appropriate timescale
        noise = randn_tensor(initial_image_latents.shape, device=torch.device(self.device), dtype=dtype)
        noisy_latents = self.pipeline.scheduler.add_noise(initial_image_latents, noise, timestep.unsqueeze(0))
        noisy_latents = self.pipeline.scheduler.scale_model_input(noisy_latents, timestep)
        noisy_latents = noisy_latents.to(device=self.device, dtype=dtype)
        ########################## Run forward pass ##########################
        noise_pred = self.pipeline.unet(
            noisy_latents,
            timestep,
            encoder_hidden_states=prompt_embeds,
            timestep_cond=None,
            cross_attention_kwargs=None,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]
        ########################## Get and save predicted image ##########################
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

    def segment_individual_image(self, image: torch.Tensor, concepts: list[str], caption: str, num_samples=1, num_inference_steps=50, **kwargs):
        # Cocnat the concepts into the prompt
        modified_caption = caption+ "," + ", ".join([f"a {concept}" for concept in concepts])
        # Run the forward pass with daam trace wrapper
        concept_heatmaps = []
        if num_samples > 1:
            timesteps = [49 for _ in range(num_samples)]
            # timesteps = list(range(num_samples))
        else:
            timesteps = [49]

        all_heatmaps = []
        for timestep in timesteps:
            with trace(self.pipeline) as tc:
                _ = self._model_forward_pass(
                    image, 
                    modified_caption,
                    timestep=timestep,
                    guidance_scale=7.0,
                    num_inference_steps=num_inference_steps,
                    height=512,
                    width=512,
                    dtype=torch.float32,
                    batch_size=1,
                )
                print(f"Modified Caption: {modified_caption}")
                heat_map = tc.compute_global_heat_map(prompt=modified_caption)
                concept_heatmaps = []
                # For each concept make a heatmap
                for concept in concepts:
                    concept_heat_map = heat_map.compute_word_heat_map(concept).heatmap
                    concept_heatmaps.append(concept_heat_map)
                concept_heatmaps = torch.stack(concept_heatmaps, dim=0)
                all_heatmaps.append(concept_heatmaps)

        all_heatmaps = torch.stack(all_heatmaps, dim=0)
        all_heatmaps = all_heatmaps.mean(0)

        return all_heatmaps, None