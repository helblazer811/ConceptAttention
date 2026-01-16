"""
Generate an image and produce concept heatmaps using Stable Diffusion 3.
"""
import os
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

    # Define prompt and concepts to track
    prompt = "A cat sitting on grass in a park with trees"
    concepts = ["cat", "grass", "sky", "tree"]

    print(f"Generating image for prompt: {prompt}")
    print(f"Tracking concepts: {concepts}")

    # Generate image with concept attention
    output = pipe.generate_image(
        prompt=prompt,
        concepts=concepts,
        height=1024,
        width=1024,
        num_inference_steps=4,
        guidance_scale=0.0,
    )

    # Save the generated image
    image_path = os.path.join(OUTPUT_DIR, "image.png")
    output.image.save(image_path)
    print(f"Saved generated image to {image_path}")

    # Save heatmaps for each concept
    for concept, heatmap in zip(concepts, output.concept_heatmaps):
        output_path = os.path.join(OUTPUT_DIR, f"{concept}.png")
        heatmap.save(output_path)
        print(f"Saved heatmap for '{concept}' to {output_path}")

    print("Done!")
