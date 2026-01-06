
from concept_attention.flux.image_generator import FluxGenerator

# import torch
# from diffusers import StableDiffusion3Pipeline

# pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
# pipe = pipe.to("cuda")

# for seed in range(20):

#     image = pipe(
#         "A fire breathing dragon in the sunny sky",
#         negative_prompt="",
#         num_inference_steps=28,
#         guidance_scale=7.0,
#         generator=torch.manual_seed(seed),
#     ).images[0]

#     image.save(f"results/image_{seed}.png")



if __name__ == "__main__":
    # prompt = "A dog by a cat with a ball in a grassy field. Sky in the background. Cloudy with sun."
    prompt = "A dragon on a rock with wings spread. Sun in sky. ."
    cuda = "cuda:0"
    generator = FluxGenerator(
        model_name="flux-schnell",
        device=cuda,
        offload=False,
    )

    # for seed in range(20):
    image = generator.generate_image(
        width=1024, 
        height=1024,
        num_steps=4,
        prompt=prompt,
        concepts=["dog", "tree", "ball"],
        seed=0,
        guidance=0.0,
    )

    image.save(f"data/dragon_image.png")