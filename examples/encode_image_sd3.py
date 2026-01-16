"""
Encode an existing image and produce concept heatmaps using Stable Diffusion 3.
"""
import os
from PIL import Image
from concept_attention.sd3 import ConceptAttentionSD3Pipeline

if __name__ == "__main__":
    # Create output directory
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "results", "sd3")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load the pipeline
    pipe = ConceptAttentionSD3Pipeline(
        model_name="stabilityai/stable-diffusion-3.5-large-turbo",
        device="cuda",
    )

    # Load the image to encode
    image_path = os.path.join(os.path.dirname(__file__), "..", "images", "dragon_image.png")
    if not os.path.exists(image_path):
        print(f"Image not found at {image_path}, generating one first...")
        # Generate a test image
        gen_output = pipe.generate_image(
            prompt="A fire breathing dragon on a rock",
            concepts=["dragon"],
            height=1024,
            width=1024,
            num_inference_steps=4,
            guidance_scale=0.0,
        )
        image = gen_output.image
    else:
        image = Image.open(image_path).convert("RGB")

    # Define concepts to track
    concepts = ["dragon", "rock", "sky", "fire"]
    caption = "A fire breathing dragon on a rock"

    print(f"Encoding image with caption: {caption}")
    print(f"Tracking concepts: {concepts}")

    # Encode image with concept attention
    output = pipe.encode_image(
        image=image,
        prompt=caption,
        concepts=concepts,
        height=1024,
        width=1024,
        timestep_index=-2,
    )

    # Save heatmaps for each concept
    for concept, heatmap in zip(concepts, output.concept_heatmaps):
        output_path = os.path.join(OUTPUT_DIR, f"encoded_{concept}.png")
        heatmap.save(output_path)
        print(f"Saved heatmap for '{concept}' to {output_path}")

    print("Done!")
