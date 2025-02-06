"""
    This baseline just returns heatmaps as the raw cross attentions.
"""
from concept_attention.flux.src.flux.sampling import prepare, unpack
import torch
import einops
import PIL

from concept_attention.image_generator import FluxGenerator
from concept_attention.segmentation import SegmentationAbstractClass, add_noise_to_image, encode_image
from concept_attention.utils import embed_concepts, linear_normalization

class RawOutputSpaceBaseline():
    """
        This class implements the cross attention baseline. 
    """

    def __init__(
        self,
        model_name: str = "flux-schnell",
        device: str = "cuda",
        offload: bool = True,
        generator = None
    ):
        super(RawOutputSpaceBaseline, self).__init__()
        # Load up the flux generator
        if generator is None:
            self.generator = FluxGenerator(
                model_name=model_name,
                device=device,
                offload=offload,
            )
        else:
            self.generator = generator
        # Unpack the tokenizer
        self.tokenizer = self.generator.t5.tokenizer

    def __call__(
        self,
        prompt,
        concepts,
        seed=4,
        num_steps=4,
        timesteps=None,
        layers=None,
        softmax=False
    ):
        """
            Generate cross attention heatmap visualizations. 

            Args:
            - prompt: str, the prompt to generate the visualizations for
            - seed: int, the seed to use for the visualization

            Returns:
            - attention_maps: torch.Tensor, the attention maps for the prompt
            - tokens: list[str], the tokens in the prompt
            - image: torch.Tensor, the image generated by the
        """
        if timesteps is None:
            timesteps = list(range(num_steps))
        if layers is None:
            layers = list(range(19))
        # Run the image generator
        image = self.generator.generate_image(
            width=1024,
            height=1024,
            num_steps=num_steps,
            guidance=0.0,
            seed=seed,
            prompt=prompt,
            concepts=concepts
        )
        # Pull out and average the attention maps
        image_attention_outputs = []
        concept_attention_outputs = []
        for double_block in self.generator.model.double_blocks:
            image_outputs = torch.stack(
                double_block.image_output_vectors
            ).squeeze(1)
            concept_outputs = torch.stack(
                double_block.concept_output_vectors
            ).squeeze(1)
            # Clear out the layer (always same)
            double_block.clear_cached_vectors()
            # Append to the list
            image_attention_outputs.append(image_outputs)
            concept_attention_outputs.append(concept_outputs)
        # Stack layers
        image_attention_outputs = torch.stack(image_attention_outputs).to(torch.float32)
        concept_attention_outputs = torch.stack(concept_attention_outputs).to(torch.float32)
        # Apply linear normalization to concepts
        concept_attention_outputs = linear_normalization(concept_attention_outputs, dim=-2)
        # Compute the maps now 
        concept_maps = einops.einsum(
            image_attention_outputs,
            concept_attention_outputs,
            "layers timesteps pixels dims, layers timesteps concepts dims -> layers timesteps concepts pixels"
        )
        if softmax:
            concept_maps = torch.nn.functional.softmax(concept_maps, dim=-2)
        # Pull out the desired timesteps
        concept_maps = concept_maps[:, timesteps]
        # Pull out the desired layers
        concept_maps = concept_maps[layers]
        # Average over layers and timesteps
        concept_maps = einops.reduce(
            concept_maps,
            "layers timesteps concepts pixels -> concepts pixels",
            reduction="mean"
        )
        concept_maps = einops.rearrange(
            concept_maps,
            "concepts (h w) -> concepts h w",
            h=64,
            w=64
        )

        return concept_maps, image

class RawOutputSpaceSegmentationModel(SegmentationAbstractClass):

    def __init__(
        self,
        model_name: str = "flux-schnell",
        device: str = "cuda",
        offload: bool = True,
        generator=None,
    ):
        """
            Initialize the segmentation model.
        """
        super(RawOutputSpaceSegmentationModel, self).__init__()
        if generator is not None:
            self.generator = generator
        else:
            # Load up the flux generator
            self.generator = FluxGenerator(
                model_name=model_name,
                device=device,
                offload=offload,
            )

        self.is_schnell = "schnell" in model_name

    def segment_individual_image(
        self,
        image: PIL.Image.Image,
        concepts: list[str],
        caption: str,
        device: str = "cuda",
        offload: bool = False,
        num_samples: int = 1,
        num_steps: int = 4,
        noise_timestep: int = 2,
        seed: int = 4,
        width: int = 1024,
        height: int = 1024,
        stop_after_multimodal_attentions: bool = True,
        layers: list[int] = list(range(19)),
        timesteps: list[int] = [-1],
        normalize_concepts=True,
        softmax: bool = False,
        joint_attention_kwargs=None,
        **kwargs
    ):
        """
            Takes a real image and generates a segmentation map. 
        """
        # Encode the image into the VAE latent space
        encoded_image_without_noise = encode_image(
            image,
            self.generator.ae,
            offload=offload,
            device=device,
        )
        # Do N trials
        for i in range(num_samples):
            # Add noise to image
            encoded_image, timesteps = add_noise_to_image(
                encoded_image_without_noise,
                num_steps=num_steps,
                noise_timestep=noise_timestep,
                seed=seed + i,
                width=width,
                height=height,
                device=device,
                is_schnell=self.is_schnell,
            )
            # Now run the diffusion model once on the noisy image
            # Encode the concept vectors
            
            if offload:
                self.generator.t5, self.generator.clip = self.generator.t5.to(device), self.generator.clip.to(device)
            inp = prepare(t5=self.generator.t5, clip=self.generator.clip, img=encoded_image, prompt=caption)

            concept_embeddings, concept_ids, concept_vec = embed_concepts(
                self.generator.clip,
                self.generator.t5,
                concepts,
            )

            inp["concepts"] = concept_embeddings.to(encoded_image.device)
            inp["concept_ids"] = concept_ids.to(encoded_image.device)
            inp["concept_vec"] = concept_vec.to(encoded_image.device)
            # offload TEs to CPU, load model to gpu
            if offload:
                self.generator.t5, self.generator.clip = self.generator.t5.cpu(), self.generator.clip.cpu()
                torch.cuda.empty_cache()
                self.generator.model = self.generator.model.to(device)
            # Denoise the intermediate images
            guidance_vec = torch.full((encoded_image.shape[0],), 0.0, device=encoded_image.device, dtype=encoded_image.dtype)
            t_curr = timesteps[0]
            t_prev = timesteps[1]
            t_vec = torch.full((encoded_image.shape[0],), t_curr, dtype=encoded_image.dtype, device=encoded_image.device)
            pred = self.generator.model(
                img=inp["img"],
                img_ids=inp["img_ids"],
                txt=inp["txt"],
                txt_ids=inp["txt_ids"],
                concepts=inp["concepts"],
                concept_ids=inp["concept_ids"],
                concept_vec=inp["concept_vec"],
                null_txt=inp["null_txt"],
                null_txt_vec=inp["null_txt_vec"],
                null_txt_ids=inp["null_txt_ids"],
                y=inp["concept_vec"],
                timesteps=t_vec,
                guidance=guidance_vec,
                stop_after_multimodal_attentions=stop_after_multimodal_attentions,
                joint_attention_kwargs=joint_attention_kwargs,
            )

        if not stop_after_multimodal_attentions:
            img = inp["img"] + (t_prev - t_curr) * pred
            # decode latents to pixel space
            img = unpack(img.float(), height, width)
            with torch.autocast(device_type=self.generator.device.type, dtype=torch.bfloat16):
                img = self.generator.ae.decode(img)

            if self.generator.offload:
                self.generator.ae.decoder.cpu()
                torch.cuda.empty_cache()
            img = img.clamp(-1, 1)
            img = einops.rearrange(img[0], "c h w -> h w c")
            # reconstructed_image = PIL.Image.fromarray(img.cpu().byte().numpy())
            reconstructed_image = PIL.Image.fromarray((127.5 * (img + 1.0)).cpu().byte().numpy())
        else:
            img = None
            reconstructed_image = None
        # Decode the image 
        if offload:
            self.generator.model.cpu()
            torch.cuda.empty_cache()
            self.generator.ae.decoder.to(device)

        # Pull out the concept basis and image queries
        concept_vectors = []
        image_vectors = []
        for double_block in self.generator.model.double_blocks:
            # Target space is the cross attention space 
            image_vecs = torch.stack(
                double_block.image_output_vectors
            ).squeeze(1)
            concept_vecs = torch.stack(
                double_block.concept_output_vectors
            ).squeeze(1)
            # Clear out the layer (always same)
            double_block.clear_cached_vectors()
            # Add to list
            concept_vectors.append(concept_vecs)
            image_vectors.append(image_vecs)
        # Stack layers
        concept_vectors = torch.stack(concept_vectors).to(torch.float32)
        image_vectors = torch.stack(image_vectors).to(torch.float32)

        if layers is not None:
            # Pull out the layer index
            concept_vectors = concept_vectors[layers]
            image_vectors = image_vectors[layers]

        # Apply linear normalization to concepts
        if normalize_concepts:
            concept_vectors = linear_normalization(concept_vectors, dim=-2)

        # Check if keep_head_dim is set
        if joint_attention_kwargs is not None and joint_attention_kwargs.get("keep_head_dim", False):
            concept_heatmaps = einops.einsum(
                concept_vectors,
                image_vectors,
                "layers timesteps heads concepts dims, layers timesteps heads pixels dims -> layers timesteps heads concepts pixels"
            )
            if softmax:
                concept_heatmaps = torch.nn.functional.softmax(concept_heatmaps, dim=-2)
            concept_heatmaps = einops.reduce(
                concept_heatmaps,
                "layers timesteps heads concepts pixels -> concepts pixels",
                reduction="mean"
            )
        else:
            # Now compute the heatmap
            concept_heatmaps = einops.einsum(
                concept_vectors,
                image_vectors,
                "layers timesteps concepts dims, layers timesteps pixels dims -> layers timesteps concepts pixels"
            )

            if softmax:
                concept_heatmaps = torch.nn.functional.softmax(concept_heatmaps, dim=-2)

            concept_heatmaps = einops.reduce(
                concept_heatmaps,
                "layers timesteps concepts pixels -> concepts pixels",
                reduction="mean"
            )
            
        concept_heatmaps = einops.rearrange(
            concept_heatmaps,
            "concepts (h w) -> concepts h w",
            h=64,
            w=64
        )

        return concept_heatmaps, reconstructed_image