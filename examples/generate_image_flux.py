"""
    Producing concept heatmaps for a generated image.
"""
import os
from concept_attention import ConceptAttentionFluxPipeline

if __name__ == "__main__":
    pipeline = ConceptAttentionFluxPipeline(
        model_name="flux-schnell",
        device="cuda:0"
    )

    prompt = "A cat in a park on the grass by a tree"
    concepts = ["cat", "grass", "sky", "tree"]

    pipeline_output = pipeline.generate_image(
        prompt=prompt,
        concepts=concepts,
        width=1024,
        height=1024
    )

    image = pipeline_output.image
    concept_heatmaps = pipeline_output.concept_heatmaps
    cross_attention_heatmaps = pipeline_output.cross_attention_maps

    os.makedirs("results/flux", exist_ok=True)

    image.save("results/flux/image.png")
    for concept, concept_heatmap in zip(concepts, concept_heatmaps):
        concept_heatmap.save(f"results/flux/{concept}.png")

    for concept, cross_attention_heatmap in zip(concepts, cross_attention_heatmaps):
        cross_attention_heatmap.save(f"results/flux/cross_attention_{concept}.png")
