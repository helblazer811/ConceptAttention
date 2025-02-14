"""
    A wrapper around a flux model that generates segmentation masks for particular 
    concepts. 
"""
from abc import ABC, abstractmethod
import PIL
import torch
import numpy as np
import einops
import PIL
from torchvision import transforms
import torchvision.transforms.functional as F

from concept_attention.flux.src.flux.sampling import get_noise, get_schedule, prepare, unpack

from concept_attention.image_generator import FluxGenerator
from concept_attention.utils import embed_concepts, linear_normalization

class SegmentationAbstractClass(ABC):

    def segment_individual_image(
        self,
        image: PIL.Image.Image,
        concepts: list[str],
        caption: str,
        **kwargs
    ):
        """
            Segments an individual image
        """
        pass

    def __call__(
        self,
        images: PIL.Image.Image | list[PIL.Image.Image],
        target_concepts: list[str],
        concepts: list[str],
        captions: list[str],
        mean_value_threshold: bool = True,
        joint_attention_kwargs=None,
        apply_blur=False,
        **kwargs
    ):
        if not isinstance(images, list):
            images = [images]
        # Encode each image using the flux model
        all_coefficients, reconstructed_images, all_masks = [], [], []
        for index, image in enumerate(images):
            coefficients, reconstructed_image = self.segment_individual_image(
                image,
                concepts,
                captions[index],
                joint_attention_kwargs=joint_attention_kwargs,
                **kwargs
            )
            # Apply a blur to the coefficients
            if apply_blur:
                coefficients = F.gaussian_blur(coefficients.unsqueeze(0), kernel_size=3, sigma=1.0).squeeze()
            # Threshold each coefficient to make a set of masks
            mean_values = torch.mean(coefficients, dim=(1, 2), keepdim=True)
            masks = coefficients > mean_values
            # Check if there is a particular a target concept or not
            if target_concepts is None:
                # Return all masks
                all_masks.append(masks)
                all_coefficients.append(coefficients)
                reconstructed_images.append(reconstructed_image)
            else:
                # Binarize the coefficients to generate a segmentation mask
                target_concept_index = concepts.index(target_concepts[index])
                if mean_value_threshold:
                    mean_value = coefficients[target_concept_index].mean()
                    mask = coefficients[target_concept_index] > mean_value
                else:
                    mask = coefficients[target_concept_index] > 0.0
                target_concept_coefficients = coefficients[target_concept_index]
                mask = mask.cpu().numpy()
                target_concept_coefficients = target_concept_coefficients.detach().cpu().numpy()
                all_masks.append(mask)    
                all_coefficients.append(target_concept_coefficients)
                reconstructed_images.append(reconstructed_image)

        return all_masks, all_coefficients, reconstructed_images

def add_noise_to_image(
    encoded_image,
    num_steps=50,
    noise_timestep=49,
    seed=63,
    width=1024,
    height=1024,
    device="cuda",
    is_schnell=True,
):
    # prepare input
    x = get_noise(
        1,
        height,
        width,
        device=device,
        dtype=torch.bfloat16,
        seed=seed,
    )
    timesteps = get_schedule(
        num_steps,
        x.shape[-1] * x.shape[-2] // 4,
        shift=(not is_schnell),
    )
    t = timesteps[noise_timestep]
    timesteps = timesteps[noise_timestep:]
    x = t * x + (1.0 - t) * encoded_image.to(x.dtype)

    return x, timesteps

@torch.no_grad()
def encode_image(
    image: PIL.Image.Image,
    autoencoder: torch.nn.Module,
    offload=True,
    device="cuda",
    height=1024,
    width=1024,
):
    """
        Encodes a PIL image to the VAE latent space and adds noise to it
    """
    if isinstance(image, PIL.Image.Image):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: 2.0 * x - 1.0),
        ])
        image = transform(image)
    else:
        transform = transforms.Compose([
            transforms.Lambda(lambda x: 2.0 * x - 1.0),
        ])
        image = transform(image)
    # init_image = image.convert("RGB")
    # init_image = np.array(image)
    init_image = image
    if isinstance(init_image, np.ndarray):
        init_image = torch.from_numpy(init_image).permute(2, 0, 1).float() / 255.0
    init_image = init_image.unsqueeze(0) 
    init_image = init_image.to(device)
    init_image = torch.nn.functional.interpolate(init_image, (height, width))
    if offload:
        autoencoder.encoder.to(device)
    init_image = autoencoder.encode(init_image.to())
    if offload:
        autoencoder = autoencoder.cpu()
        torch.cuda.empty_cache()

    return init_image


@torch.no_grad()
def generate_concept_basis_and_image_representation(
    image: PIL.Image.Image,
    caption: str,
    concepts: list[str],
    noise_timestep: int | list[int] =49,
    layers=list(range(19)),
    normalize_concepts=True,
    num_steps=50,
    seed=63,
    model_name="flux-schnell",
    offload=True,
    device="cuda",
    target_space="output",
    height=1024,
    width=1024,
    generator=None,
    stop_after_multimodal_attentions=False,
    num_samples=1,
    joint_attention_kwargs=None,
    reduce_dims=True,
    **kwargs
):
    """
        Takes a real image and generates a set of concept and image vectors. 
    """
    if generator is None:
        # Load up the model
        generator = FluxGenerator(
            model_name,
            device, 
            offload=offload
        )
    else:
        model_name = generator.model_name
    # Encode the image into the VAE latent space
    encoded_image_without_noise = encode_image(
        image,
        generator.ae,
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
            is_schnell=False,
        )
        # Now run the diffusion model once on the noisy image
        # Encode the concept vectors
        
        if offload:
            generator.t5, generator.clip = generator.t5.to(device), generator.clip.to(device)
        inp = prepare(t5=generator.t5, clip=generator.clip, img=encoded_image, prompt=caption)

        concept_embeddings, concept_ids, concept_vec = embed_concepts(
            generator.clip,
            generator.t5,
            concepts,
        )

        inp["concepts"] = concept_embeddings.to(encoded_image.device)
        inp["concept_ids"] = concept_ids.to(encoded_image.device)
        inp["concept_vec"] = concept_vec.to(encoded_image.device)
        # offload TEs to CPU, load model to gpu
        if offload:
            generator.t5, generator.clip = generator.t5.cpu(), generator.clip.cpu()
            torch.cuda.empty_cache()
            generator.model = generator.model.to(device)
        # Denoise the intermediate images
        guidance_vec = torch.full((encoded_image.shape[0],), 0.0, device=encoded_image.device, dtype=encoded_image.dtype)
        t_curr = timesteps[0]
        t_prev = timesteps[1]
        t_vec = torch.full((encoded_image.shape[0],), t_curr, dtype=encoded_image.dtype, device=encoded_image.device)
        pred = generator.model(
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
            joint_attention_kwargs=joint_attention_kwargs
        )

    if not stop_after_multimodal_attentions:
        if offload:
            generator.model.cpu()
            torch.cuda.empty_cache()
            generator.ae.decoder.to(pred.device)

        img = inp["img"] + (t_prev - t_curr) * pred
        # decode latents to pixel space
        img = unpack(img.float(), height, width)
        with torch.autocast(device_type=generator.device.type, dtype=torch.bfloat16):
            img = generator.ae.decode(img)

        if generator.offload:
            generator.ae.decoder.cpu()
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
        generator.model.cpu()
        torch.cuda.empty_cache()
        generator.ae.decoder.to(device)

    # Pull out the concept basis and image queries
    concept_vectors = []
    image_vectors = []
    for double_block in generator.model.double_blocks:
        if target_space == "output":
            image_vecs = torch.stack(
                double_block.image_output_vectors
            ).squeeze(1)
            concept_vecs = torch.stack(
                double_block.concept_output_vectors
            ).squeeze(1)
        elif target_space == "cross_attention":
            image_vecs = torch.stack(
                double_block.image_query_vectors
            ).squeeze(1)
            concept_vecs = torch.stack(
                double_block.concept_key_vectors
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

    if reduce_dims:
        if len(image_vectors.shape) == 4:
            image_vectors = einops.rearrange(
                image_vectors,
                "layers time patches d -> patches (layers time d)",
            )
            concept_vectors = einops.rearrange(
                concept_vectors,
                "layers time concepts d -> concepts (layers time d)"
            )
        else:
            image_vectors = einops.rearrange(
                image_vectors,
                "layers time heads patches d -> patches (layers time heads d)",
            )
            concept_vectors = einops.rearrange(
                concept_vectors,
                "layers time heads concepts d -> concepts (layers time heads d)"
            )
    
    return image_vectors, concept_vectors, reconstructed_image
