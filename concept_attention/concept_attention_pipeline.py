"""
    Wrapper pipeline for concept attention. 
"""
from dataclasses import dataclass
import PIL
import numpy as np
import matplotlib.pyplot as plt
from concept_attention.flux.src.flux.sampling import prepare
from concept_attention.segmentation import add_noise_to_image, encode_image
from concept_attention.utils import embed_concepts, linear_normalization
import torch
import einops
from tqdm import tqdm

from concept_attention.binary_segmentation_baselines.raw_cross_attention import RawCrossAttentionBaseline, RawCrossAttentionSegmentationModel
from concept_attention.binary_segmentation_baselines.raw_output_space import RawOutputSpaceBaseline, RawOutputSpaceSegmentationModel
from concept_attention.image_generator import FluxGenerator

@dataclass
class ConceptAttentionPipelineOutput():
    image: PIL.Image.Image | np.ndarray
    concept_heatmaps: list[PIL.Image.Image]
    cross_attention_maps: list[PIL.Image.Image]


def compute_heatmaps_from_vectors(
    image_vectors,
    concept_vectors,
    layer_indices: list[int],
    timesteps: list[int] = list(range(4)),
    softmax: bool = True,
    normalize_concepts: bool = False
):
    """
        Accepts image vectors and concept vectors. These can be from cross attentions or attention outputs.  
    """
    # Check if there are heads in the input 
    if len(image_vectors.shape) == 6: 
        # Collapse the had dimension
        image_vectors = einops.rearrange(
            image_vectors,
            "time layers batch head patches dim -> time layers batch patches (head dim)"
        )
        concept_vectors = einops.rearrange(
            concept_vectors,
            "time layers batch head concepts dim -> time layers batch concepts (head dim)"
        )


    # Apply linear normalization to concepts
    if normalize_concepts:
        concept_vectors = linear_normalization(concept_vectors, dim=-2)

    # Compute heatmaps 
    heatmaps = einops.einsum(
        image_vectors, 
        concept_vectors,
        "time layers batch patches dim, time layers batch concepts dim -> time layers batch concepts patches",
    )

    # Apply softmax
    if softmax:
        heatmaps = torch.nn.functional.softmax(heatmaps, dim=-2)
    # Pull out the timesteps and layers
    heatmaps = heatmaps[timesteps]
    heatmaps = heatmaps[:, layer_indices]
    # Average over the heatmaps
    heatmaps = einops.reduce(
        heatmaps,
        "time layers batch concepts patches -> batch concepts patches",
        reduction="mean"
    )
    heatmaps = einops.rearrange(
        heatmaps,
        "batch concepts (h w) -> batch concepts h w",
        h=64,
        w=64
    )

    return heatmaps

class ConceptAttentionFluxPipeline():
    """
        This is an object that allows you to generate images with flux, and
        'encode' images with flux.  
    """

    def __init__(
        self, 
        model_name: str = "flux-schnell", 
        offload_model=False,
        device="cuda:0"
    ):
        self.model_name = model_name
        self.offload_model = offload_model
        # Load the generator
        self.flux_generator = FluxGenerator(
            model_name=model_name,
            offload=offload_model,
            device=device
        )

    @torch.no_grad()
    def generate_image(
        self, 
        prompt: str,
        concepts: list[str],
        width: int = 1024,
        height: int = 1024,
        return_cross_attention = False,
        layer_indices = list(range(15, 19)),
        return_pil_heatmaps = True,
        seed: int = 0,
        num_inference_steps: int = 4,
        guidance: float = 0.0,
        timesteps=None,
        softmax: bool = True,
        cmap="plasma"
    ) -> ConceptAttentionPipelineOutput:
        """
            Generate an image with flux, given a list of concepts.
        """
        assert return_cross_attention is False, "Not supported yet"
        assert all([layer_index >= 0 and layer_index < 19 for layer_index in layer_indices]), "Invalid layer index"
        assert height == width, "Height and width must be the same for now"

        if timesteps is None:
            timesteps = list(range(num_inference_steps))
        # Run the raw output space object
        image, concept_attention_dict = self.flux_generator.generate_image(
            width=width,
            height=height,
            prompt=prompt,
            num_steps=num_inference_steps,
            concepts=concepts,
            seed=seed,
            guidance=guidance,
        )
        
        cross_attention_maps = compute_heatmaps_from_vectors(
            concept_attention_dict["cross_attention_image_vectors"],
            concept_attention_dict["cross_attention_concept_vectors"],
            layer_indices=layer_indices,
            timesteps=timesteps,
            softmax=softmax
        )
        # Compute concept the heatmaps
        concept_heatmaps = compute_heatmaps_from_vectors(
            concept_attention_dict["output_space_image_vectors"],
            concept_attention_dict["output_space_concept_vectors"],
            layer_indices=layer_indices,
            timesteps=timesteps,
            softmax=softmax
        )

        concept_heatmaps = concept_heatmaps.to(torch.float32).detach().cpu().numpy()[0]
        cross_attention_maps = cross_attention_maps.to(torch.float32).detach().cpu().numpy()[0]
        # Convert the torch heatmaps to PIL images.
        if return_pil_heatmaps:
            concept_heatmaps_min = concept_heatmaps.min()
            concept_heatmaps_max = concept_heatmaps.max()
            # Convert to a matplotlib color scheme
            colored_heatmaps = []
            for concept_heatmap in concept_heatmaps:
                concept_heatmap = (concept_heatmap - concept_heatmaps_min) / (concept_heatmaps_max - concept_heatmaps_min)
                colored_heatmap = plt.get_cmap(cmap)(concept_heatmap)
                rgb_image = (colored_heatmap[:, :, :3] * 255).astype(np.uint8)
                colored_heatmaps.append(rgb_image)

            concept_heatmaps = [PIL.Image.fromarray(concept_heatmap) for concept_heatmap in colored_heatmaps]

            cross_attention_min = cross_attention_maps.min()
            cross_attention_max = cross_attention_maps.max()
            colored_cross_attention_maps = []
            for cross_attention_map in cross_attention_maps:
                cross_attention_map = (cross_attention_map - cross_attention_min) / (cross_attention_max - cross_attention_min)
                colored_cross_attention_map = plt.get_cmap(cmap)(cross_attention_map)
                rgb_image = (colored_cross_attention_map[:, :, :3] * 255).astype(np.uint8)
                colored_cross_attention_maps.append(rgb_image)

            cross_attention_maps = [PIL.Image.fromarray(cross_attention_map) for cross_attention_map in colored_cross_attention_maps]

        return ConceptAttentionPipelineOutput(
            image=image,
            concept_heatmaps=concept_heatmaps,
            cross_attention_maps=cross_attention_maps
        )

    def encode_image(
        self,
        image: PIL.Image.Image,
        concepts: list[str],
        prompt: str = "", # Optional
        width: int = 1024,
        height: int = 1024,
        layer_indices = list(range(15, 19)),
        num_samples: int = 1,
        num_steps: int = 4,
        noise_timestep: int = 2,
        device: str = "cuda:0",
        return_pil_heatmaps: bool = True,
        seed: int = 0,
        cmap="plasma",
        stop_after_multi_modal_attentions=True,
        softmax=True
    ) -> ConceptAttentionPipelineOutput:
        """
            Encode an image with flux, given a list of concepts.
        """
        assert all([layer_index >= 0 and layer_index < 19 for layer_index in layer_indices]), "Invalid layer index"
        assert height == width, "Height and width must be the same for now"
        print("Encoding image")

        # Encode the image into the VAE latent space
        encoded_image_without_noise = encode_image(
            image,
            self.flux_generator.ae,
            offload=self.flux_generator.offload,
            device=device,
        )
        # Do N trials
        combined_concept_attention_dict = {
            "cross_attention_image_vectors": [],
            "cross_attention_concept_vectors": [],
            # "cross_attention_maps": [],
            "output_space_image_vectors": [],
            "output_space_concept_vectors": [],
        }
        print("Sampling")
        for i in tqdm(range(num_samples)):
            # Add noise to image
            encoded_image, timesteps = add_noise_to_image(
                encoded_image_without_noise,
                num_steps=num_steps,
                noise_timestep=noise_timestep,
                seed=seed + i,
                width=width,
                height=height,
                device=device,
                is_schnell=self.flux_generator.is_schnell,
            )
            # Now run the diffusion model once on the noisy image
            # Encode the concept vectors
            
            if self.flux_generator.offload:
                self.flux_generator.t5, self.flux_generator.clip = self.flux_generator.t5.to(device), self.flux_generator.clip.to(device)
            inp = prepare(t5=self.flux_generator.t5, clip=self.flux_generator.clip, img=encoded_image, prompt=prompt)

            concept_embeddings, concept_ids, concept_vec = embed_concepts(
                self.flux_generator.clip,
                self.flux_generator.t5,
                concepts,
            )

            inp["concepts"] = concept_embeddings.to(encoded_image.device)
            inp["concept_ids"] = concept_ids.to(encoded_image.device)
            inp["concept_vec"] = concept_vec.to(encoded_image.device)
            # offload TEs to CPU, load model to gpu
            if self.flux_generator.offload:
                self.flux_generator.t5, self.flux_generator.clip = self.flux_generator.t5.cpu(), self.flux_generator.clip.cpu()
                torch.cuda.empty_cache()
                self.flux_generator.model = self.flux_generator.model.to(device)
            # Denoise the intermediate images
            guidance_vec = torch.full((encoded_image.shape[0],), 0.0, device=encoded_image.device, dtype=encoded_image.dtype)
            t_curr = timesteps[0]
            t_prev = timesteps[1]
            t_vec = torch.full((encoded_image.shape[0],), t_curr, dtype=encoded_image.dtype, device=encoded_image.device)
            _, concept_attention_dict = self.flux_generator.model(
                img=inp["img"],
                img_ids=inp["img_ids"],
                txt=inp["txt"],
                txt_ids=inp["txt_ids"],
                concepts=inp["concepts"],
                concept_ids=inp["concept_ids"],
                concept_vec=inp["concept_vec"],
                y=inp["concept_vec"],
                timesteps=t_vec,
                guidance=guidance_vec,
                stop_after_multimodal_attentions=stop_after_multi_modal_attentions, # Always true for the demo
                joint_attention_kwargs=None,
            )

            for key in combined_concept_attention_dict.keys():
                combined_concept_attention_dict[key].append(concept_attention_dict[key])

        # Pull out the concept and image vectors from each block
        for key in combined_concept_attention_dict.keys():
            combined_concept_attention_dict[key] = torch.stack(combined_concept_attention_dict[key]).squeeze(1)

        # Compute the heatmaps
        concept_heatmaps = compute_heatmaps_from_vectors(
            combined_concept_attention_dict["output_space_image_vectors"],
            combined_concept_attention_dict["output_space_concept_vectors"],
            layer_indices=layer_indices,
            timesteps=timesteps,
            softmax=softmax
        )

        cross_attention_maps = compute_heatmaps_from_vectors(
            combined_concept_attention_dict["cross_attention_image_vectors"],
            combined_concept_attention_dict["cross_attention_concept_vectors"],
            layer_indices=layer_indices,
            timesteps=timesteps,
            softmax=softmax
        )

        concept_heatmaps = concept_heatmaps.to(torch.float32).detach().cpu().numpy()[0]
        cross_attention_maps = cross_attention_maps.to(torch.float32).detach().cpu().numpy()[0]
        # Convert the torch heatmaps to PIL images.
        if return_pil_heatmaps:
            concept_heatmaps_min = concept_heatmaps.min()
            concept_heatmaps_max = concept_heatmaps.max()
            # Convert to a matplotlib color scheme
            colored_heatmaps = []
            for concept_heatmap in concept_heatmaps:
                concept_heatmap = (concept_heatmap - concept_heatmaps_min) / (concept_heatmaps_max - concept_heatmaps_min)
                colored_heatmap = plt.get_cmap(cmap)(concept_heatmap)
                rgb_image = (colored_heatmap[:, :, :3] * 255).astype(np.uint8)
                colored_heatmaps.append(rgb_image)

            concept_heatmaps = [PIL.Image.fromarray(concept_heatmap) for concept_heatmap in colored_heatmaps]

            cross_attention_min = cross_attention_maps.min()
            cross_attention_max = cross_attention_maps.max()
            colored_cross_attention_maps = []
            for cross_attention_map in cross_attention_maps:
                cross_attention_map = (cross_attention_map - cross_attention_min) / (cross_attention_max - cross_attention_min)
                colored_cross_attention_map = plt.get_cmap(cmap)(cross_attention_map)
                rgb_image = (colored_cross_attention_map[:, :, :3] * 255).astype(np.uint8)
                colored_cross_attention_maps.append(rgb_image)

            cross_attention_maps = [PIL.Image.fromarray(cross_attention_map) for cross_attention_map in colored_cross_attention_maps]

        return ConceptAttentionPipelineOutput(
            image=image,
            concept_heatmaps=concept_heatmaps,
            cross_attention_maps=cross_attention_maps
        )

