from PIL import Image
from concept_attention.sd3.sd3 import SD3SegmentationModel
import matplotlib.pyplot as plt
import os


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Encode image with SD3 Concept Attention"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run the model on (e.g., 'cuda' or 'cpu')",
    )
    parser.add_argument(
        "--concepts",
        type=str,
        nargs="+",
        default=["cat", "leaf", "grass"],
        help="List of concepts to generate saliency maps for",
    )
    parser.add_argument(
        "--caption",
        type=str,
        default="A photo of a cat in grass with leaves",
        help="Caption describing the image",
    )
    parser.add_argument(
        "--image",
        type=str,
        default="image.png",
        help="Path to the input image",
    )
    args = parser.parse_args()

    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)

    # Load the segmentation model
    segmentation_model = SD3SegmentationModel(
        device=args.device,
        mode="concept_attention",
    )
    # Load the test image
    image = Image.open(args.image).convert("RGB")

    # Run segmentation to get concept heatmaps
    print(f"Generating saliency maps for concepts: {args.concepts}")
    concept_heatmaps, _ = segmentation_model.segment_individual_image(
        image=image,
        concepts=args.concepts,
        caption=args.caption,
        decode_image=False,
    )

    # Save saliency maps for each concept
    for i, concept in enumerate(args.concepts):
        # Get the heatmap for this concept
        heatmap = concept_heatmaps[i].numpy()

        # Create and save the figure with no padding or axes
        fig = plt.figure(figsize=(8, 8), frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(heatmap, cmap='inferno', interpolation='none')

        # Save the figure
        output_path = os.path.join("results", f"saliency_{concept}.png")
        plt.savefig(output_path, dpi=150)
        plt.close()

        print(f"Saved saliency map for '{concept}' to {output_path}")

    print(f"\nAll saliency maps saved to results/ directory")
