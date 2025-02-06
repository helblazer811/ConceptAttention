"""
    Producing concept heatmaps for a generated image. 
"""
from concept_attention import ConceptAttentionFluxPipeline

pipeline = ConceptAttentionFluxPipeline(
    model_name="flux-schnell",
    device="cuda:0"
)

prompt = "A cat holding a sign that says hello world"
concepts = ["cat", "sign", "sky", "tree"]

pipeline_output = pipeline.generate_image(
    prompt=prompt,
    concepts=concepts,
    width=1024,
    height=1024,
)

image = pipeline_output.image
concept_heatmaps = pipeline_output.concept_heatmaps

image.save("image.png")
for concept, concept_heatmap in zip(concepts, concept_heatmaps):
    concept_heatmap.save(f"{concept}.png")
