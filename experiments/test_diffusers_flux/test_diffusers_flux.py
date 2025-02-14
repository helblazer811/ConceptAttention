"""

concept_attention_kwargs cases:
- case 1: concept_attention_kwargs is None
    - Don't do concept attention 
- case 2: concept_attention_kwargs is not None
    - Mandate that concept_attention_layers, timesteps, and concepts are in concept_attention_kwargs and are not None

"""
import torch
from matplotlib import cm
from diffusers import FluxPipeline

from concept_attention.diffusers.flux import FluxWithConceptAttentionPipeline, FluxTransformer2DModelWithConceptAttention

if __name__ == "__main__":
    transformer = FluxTransformer2DModelWithConceptAttention.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        torch_dtype=torch.bfloat16,
        subfolder="transformer"
    )
    pipe = FluxWithConceptAttentionPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell", 
        transformer=transformer,
        torch_dtype=torch.bfloat16
    )
    pipe.enable_model_cpu_offload()

    prompt = "A cat on the grass"
    out = pipe(
        prompt=prompt,
        guidance_scale=0.,
        height=1024,
        width=1024,
        num_inference_steps=4,
        max_sequence_length=256,
        concept_attention_kwargs={
            "layers": list(range(18)),
            "timesteps": list(range(3, 4)),
            "concepts": ["cat", "grass", "sky", "background", "dog"]
        },
        # output_type="latent"
    )
    image = out.images[0]
    concept_attention_maps = out.concept_attention_maps[0]
    image.save("image.png")
    # Pull out and save the concept attention maps
    for i, attention_map in enumerate(concept_attention_maps):
        attention_map.save(f"images/attention_map_{i}.png")
