import PIL
import os

from concept_attention import ConceptAttentionFluxPipeline

def test_concept_heatmaps():
    """
    Test concept heatmap generation using the pipeline's encode_image method with multiple images.
    Saves heatmaps to results/test_encode_concepts_parallel directory.
    """
    # Use the same results directory
    results_dir = "results/test_encode_concepts_parallel"
    os.makedirs(results_dir, exist_ok=True)
    
    print("\n=== Testing Multiple Image Concept Heatmap Generation ===")
    
    # Initialize pipeline
    pipeline = ConceptAttentionFluxPipeline(
        model_name="flux-schnell",    
        device="cuda:0"
    )

    # Load multiple images
    try:
        image_1 = PIL.Image.open("results/image.png")
        print("Loaded image 1: results/image.png")
    except FileNotFoundError:
        print("Error: results/image.png not found")
        return
        
    try:
        image_2 = PIL.Image.open("results/cat.png")
        print("Loaded image 2: results/cat.png")
    except FileNotFoundError:
        print("Warning: results/cat.png not found, using image.png for both")
        image_2 = image_1
    
    # Create lists of images, concepts, and prompts
    images = [image_1, image_2]
    concepts_lists = [
        ["dragon", "rock", "sky", "sun", "clouds"],
        ["cat", "tree", "grass", "building", "sky"]
    ]
    prompts = [
        "A fire breathing dragon in a rocky landscape with sky and clouds in the sun.",
        "A cat sitting near a tree on grass with buildings in the background under the sky."
    ]

    # Run the pipeline encode_image method with multiple inputs
    print("Running pipeline encode_image with multiple images...")
    pipeline_output = pipeline.encode_image(
        image=images,
        concepts=concepts_lists,
        prompt=prompts,
        width=1024,
        height=1024,
    )

    concept_heatmaps = pipeline_output.concept_heatmaps

    # Save concept heatmaps for each image
    print("Saving concept heatmaps...")
    for img_idx, (image, concepts, prompt) in enumerate(zip(images, concepts_lists, prompts)):
        print(f"\nProcessing image {img_idx + 1}: {prompt}")
        
        # Create subdirectory for this image
        img_results_dir = os.path.join(results_dir, f"image_{img_idx + 1}")
        os.makedirs(img_results_dir, exist_ok=True)
        
        # Save heatmaps for this image's concepts
        img_heatmaps = concept_heatmaps[img_idx]
        for concept_idx, (concept, concept_heatmap) in enumerate(zip(concepts, img_heatmaps)):
            heatmap_path = os.path.join(img_results_dir, f"{concept}_heatmap.png")
            concept_heatmap.save(heatmap_path)
            print(f"  Saved heatmap: {heatmap_path}")
        
        # Save the original image used for heatmaps
        original_heatmap_path = os.path.join(img_results_dir, "input_image.png")
        image.save(original_heatmap_path)
        print(f"  Saved input image: {original_heatmap_path}")
    
    print(f"\n✅ All concept heatmaps saved to {results_dir}")


if __name__ == "__main__":
    test_concept_heatmaps()
