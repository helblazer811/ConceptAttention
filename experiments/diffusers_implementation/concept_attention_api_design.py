import torch
from concept_attention.diffusers_concept_attention import ConceptAttentionFluxPipeline

if __name__ == "__main__":

    pipe = ConceptAttentionFluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()

    prompt = "A cat holding a sign that says hello world"
    concepts = ["cat", "sign", "sky", "tree"]
    out = pipe(
        prompt=prompt,
        concepts=concepts,
        guidance_scale=0.0,
        height=768,
        width=1360,
        num_inference_steps=4,
        max_sequence_length=256,
    )
    image = out.images[0]
    concepts = out.concepts # list[PIL.Image.Image]
    out.save("image.png")