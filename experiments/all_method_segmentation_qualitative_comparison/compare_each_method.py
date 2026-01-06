"""
    Run each binary segmentation method on a single image from the ImageNet Segmentation dataset
    and get soft and hard masks for them. Save all of the images. 
"""

from PIL import Image
import einops
import matplotlib.pyplot as plt
import os

from concept_attention.binary_segmentation_baselines.chefer_clip_vit_baselines import CheferAttentionGradCAMSegmentationModel, CheferFullLRPSegmentationModel, CheferLRPSegmentationModel, CheferLastLayerAttentionSegmentationModel, CheferLastLayerLRPSegmentationModel, CheferRolloutSegmentationModel, CheferTransformerAttributionSegmentationModel
from concept_attention.binary_segmentation_baselines.clip_text_span_baseline import CLIPTextSpanSegmentationModel
from concept_attention.binary_segmentation_baselines.daam_sdxl import DAAMStableDiffusionXLSegmentationModel
from concept_attention.binary_segmentation_baselines.dino import DINOSegmentationModel
from concept_attention.binary_segmentation_baselines.raw_cross_attention import RawCrossAttentionSegmentationModel
from concept_attention.binary_segmentation_baselines.raw_output_space import RawOutputSpaceSegmentationModel
from concept_attention.binary_segmentation_baselines.raw_value_space import RawValueSpaceSegmentationModel
from concept_attention.flux.image_generator import FluxGenerator

from torchvision import transforms

from new_paper_experiments.imagenet_segmentation.data_processing import ImagenetSegmentation

def load_model(segmentation_model: str, device: str):

    # Load up the model
    target_space = None
    if segmentation_model == "RawCrossAttention":
        segmentation_model = RawCrossAttentionSegmentationModel(
            model_name="flux-schnell",
            offload=False,
            device=device,
        )
    elif segmentation_model == "RawValueSpace":
        segmentation_model = RawValueSpaceSegmentationModel(
            model_name="flux-schnell",
            offload=False,
            device=device,
        )
    elif segmentation_model == "RawOutputSpace":
        segmentation_model = RawOutputSpaceSegmentationModel(
            model_name="flux-schnell",
            offload=False,
            device=device,
        )
    elif segmentation_model == "CheferLRP":
        segmentation_model = CheferLRPSegmentationModel(device=device)
    elif segmentation_model == "CheferRollout":
        segmentation_model = CheferRolloutSegmentationModel(device=device)
    elif segmentation_model == "CheferLastLayerAttention":
        segmentation_model = CheferLastLayerAttentionSegmentationModel(device=device)
    elif segmentation_model == "CheferAttentionGradCAM":
        segmentation_model = CheferAttentionGradCAMSegmentationModel(device=device)
    elif segmentation_model == "CheferTransformerAttribution":
        segmentation_model = CheferTransformerAttributionSegmentationModel(device=device)
    elif segmentation_model == "CheferFullLRP":
        segmentation_model = CheferFullLRPSegmentationModel(device=device)
    elif segmentation_model == "CheferLastLayerLRP":
        segmentation_model = CheferLastLayerLRPSegmentationModel(device=device)
    elif segmentation_model == "DAAMSDXL":
        segmentation_model = DAAMStableDiffusionXLSegmentationModel(device=device)
    elif segmentation_model == "CLIPTextSpan":
        segmentation_model = CLIPTextSpanSegmentationModel(device=device)
    elif segmentation_model == "DINO":
        segmentation_model = DINOSegmentationModel(device=device)
    else:
        raise ValueError(f"Segmentation model {segmentation_model} not recognized.")

    return segmentation_model

if __name__ == "__main__":

    model_names = [
        "RawCrossAttention",
        "RawOutputSpace",
        "RawValueSpace",
        "CheferLRP",
        "CheferRollout",
        "CheferLastLayerAttention",
        "CheferAttentionGradCAM",
        "CheferTransformerAttribution",
        "CheferFullLRP",
        "CheferLastLayerLRP",
        "DAAMSDXL",
        "CLIPTextSpan",
        "DINO"
    ]

    # # Make transforms for the images and labels (masks)
    image_transforms = transforms.Compose([
        # transforms.Resize((224, 224), Image.NEAREST),
        transforms.Resize((224, 224), Image.NEAREST),
        transforms.ToTensor(),
    ])

    dataset = ImagenetSegmentation(
        directory="../imagenet_segmentation/data/imagenet_segmentation",
    )
    
    # image_index = 428
    image_index = 466
    img, target, name = dataset[image_index]

    os.makedirs(f"results/{image_index}", exist_ok=True)

    # Save the image and gt

    image = image_transforms(img)
    image = transforms.ToPILImage()(image)

    image.save(f"results/{image_index}/image.png")

    target = transforms.Resize((64, 64), Image.NEAREST)(target)

    # Save the gt with matplotlib
    plt.figure()
    plt.imshow(target, cmap="viridis")
    plt.axis("off")
    plt.savefig(f"results/{image_index}/gt.png", bbox_inches="tight", pad_inches=0)

    img = image_transforms(img).to("cuda:0")
    segmentation_model = None

    for model_name in model_names:
        del segmentation_model
        # Load the model 
        segmentation_model = load_model(model_name, device="cuda:0")
        # Do the segmentation
        mask, coefficients, _ = segmentation_model(
            img,
            target_concepts=["dog"],
            concepts=["dog", "background", "fence"],
            captions=["a dog"],
            device="cuda:0",
            softmax=True,
            layers=list(range(15, 19)),
            num_samples=5
        )
        # Plot the coefficients
        plt.figure()
        plt.imshow(coefficients[0], cmap="viridis")
        plt.axis("off")
        plt.savefig(f"results/{image_index}/{model_name}.png", bbox_inches="tight", pad_inches=0)

        plt.figure()
        plt.imshow(mask[0], cmap="viridis")
        plt.axis("off")
        plt.savefig(f"results/{image_index}/{model_name}_mask.png", bbox_inches="tight", pad_inches=0)