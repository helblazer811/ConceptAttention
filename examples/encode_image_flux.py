"""
    Encode a real image using the pipeline.
"""
import os
import PIL
from concept_attention import ConceptAttentionFluxPipeline

if __name__ == "__main__":
    pipeline = ConceptAttentionFluxPipeline(
        model_name="flux-schnell",
        device="cuda:0"
    )

    image = PIL.Image.open("images/dragon_image.png")
    concepts = ["dragon", "rock", "sky", "sun", "clouds"]

    pipeline_output = pipeline.encode_image(
        image=image,
        concepts=concepts,
        prompt="A fire breathing dragon.",
        width=1024,
        height=1024,
    )

    concept_heatmaps = pipeline_output.concept_heatmaps

    os.makedirs("results/flux", exist_ok=True)

    for concept, concept_heatmap in zip(concepts, concept_heatmaps):
        concept_heatmap.save(f"results/flux/{concept}.png")
