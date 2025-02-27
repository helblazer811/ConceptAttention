from concept_attention.concept_attention_pipeline import ConceptAttentionFluxPipeline
from PIL import Image

if __name__ == "__main__":
    pipeline = ConceptAttentionFluxPipeline(
        model_name="flux-schnell", 
        offload_model=True
    ) # , device="cuda:0") # , offload_model=True)

    image = Image.open("image.png").convert("RGB")

    outputs = pipeline.encode_image(
        image,
        concepts=["animal", "background"]
    )
    concept_attention_maps = outputs.concept_heatmaps

    concepts = ["animal", "background"]
    for concept, attention_map in zip(concepts, concept_attention_maps):
        attention_map.save(f"{concept}_attention_map.png")
