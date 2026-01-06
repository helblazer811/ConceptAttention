"""
    Producing concept heatmaps for a generated image using Flux 2.
"""
from concept_attention.flux2 import ConceptAttentionFlux2Pipeline

pipeline = ConceptAttentionFlux2Pipeline(
    model_name="flux.2-dev",
    device="cuda:0"
)

prompt = "A cat sitting on grass in a park with trees"
concepts = ["cat", "grass", "sky", "tree"]

pipeline_output = pipeline.generate_image(
    prompt=prompt,
    concepts=concepts,
    width=2048,
    height=2048,
)

image = pipeline_output.image
concept_heatmaps = pipeline_output.concept_heatmaps
cross_attention_heatmaps = pipeline_output.cross_attention_maps

import os
os.makedirs("results/flux2", exist_ok=True)

image.save("results/flux2/image.png")
for concept, concept_heatmap in zip(concepts, concept_heatmaps):
    concept_heatmap.save(f"results/flux2/{concept}.png")

for concept, cross_attention_heatmap in zip(concepts, cross_attention_heatmaps):
    cross_attention_heatmap.save(f"results/flux2/cross_attention_{concept}.png")
