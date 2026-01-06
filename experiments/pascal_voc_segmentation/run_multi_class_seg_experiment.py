"""
    Here the experiment is to see if we can get good performance on the 
    zero-shot ImageNet segmentation task. 
"""
import torch
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
import os
import nltk

from concept_attention.binary_segmentation_baselines.clip_text_span_baseline import CLIPTextSpanSegmentationModel
from concept_attention.binary_segmentation_baselines.daam_sdxl import DAAMStableDiffusionXLSegmentationModel
nltk.download('wordnet')
from PIL import Image
import argparse

from new_paper_experiments.pascal_voc_segmentation.multi_class_segmentation import FluxMultiClassSegmentation
from concept_attention.flux.image_generator import FluxGenerator
from concept_attention.utils import batch_intersection_union, batch_pix_accuracy, get_ap_scores
from concept_attention.binary_segmentation_baselines.chefer_vit_explainability.data.VOC import VOCSegmentation

# from data_processing import ImagenetSegmentation

def map_predictions_to_voc_class_indices(
    predictions: torch.Tensor,
    present_classes: list[str],
    background_concepts: list[str],
):
    # Assume order of predictions is background concepts then present classes
    # First set all background concept predictions to zero
    predictions[predictions < len(background_concepts)] = 0
    # Now map the present classes to the appropriate indices
    for index, present_class in enumerate(present_classes):
        predictions[predictions == len(background_concepts) + index] = VOCSegmentation.CLASSES_NAMES.index(present_class)

    return predictions
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--segmentation_model", 
        type=str, 
        # default="RawCrossAttention",
        default="CLIPTextSpan",
        # default="CheferAttentionGradCAM",
        choices=[
            "RawOutputSpace", 
            # "RawValueSpace", 
            "RawCrossAttention",
            "CLIPTextSpan",
            "DAAMSDXL",
            # "CheferLRP",
            # "CheferRollout",
            # "CheferLastLayerAttention",
            # "CheferAttentionGradCAM",
            # "CheferTransformerAttribution",
            # "CheferFullLRP",
            # "CheferLastLayerLRP",
        ]
    )
    # parser.add_argument("--target_space", type=str, default="output")
    parser.add_argument("--device", type=str, default="cuda:2")
    parser.add_argument("--normalize_concepts", default=True)
    parser.add_argument("--softmax", action="store_true", default=False)
    parser.add_argument("--offload", default=False)
    parser.add_argument("--concept_cross_attention", action="store_true", default=True)
    parser.add_argument("--concept_self_attention", action="store_true", default=True)
    parser.add_argument("--downscale_for_eval", action="store_true")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--model_name", type=str, default="flux-schnell")
    parser.add_argument("--num_steps", type=int, default=4)
    parser.add_argument("--noise_timestep", type=int, default=3)
    parser.add_argument("--layers", type=int, nargs="+", default=list(range(14, 18)))
    parser.add_argument("--background_concepts", type=list, default=["background", "floor", "grass", "tree", "sky"]) #, "floor", "tree", "person", "grass", "face"])
    # parser.add_argument("--resize_ablation", action="store_true")
    # parser.add_argument("--image_save_dir", type=str, default="results/segmentation_predictions/multi_class_concept_attention")
    parser.add_argument("--image_save_dir", type=str, default="results/multi_class_segmentation/segmentations/raw_cross_attention")

    args = parser.parse_args()

    if not os.path.exists(args.image_save_dir):
        os.makedirs(args.image_save_dir)

    target_space = None
    if args.segmentation_model == "RawCrossAttention" or args.segmentation_model == "RawOutputSpace" or args.segmentation_model == "RawValueSpace":
        # Load up the model
        generator = FluxGenerator(
            model_name=args.model_name,
            offload=args.offload,
            device=args.device,
        )
        segmentation_model = FluxMultiClassSegmentation(
            generator=generator,
        )

        if args.segmentation_model == "RawCrossAttention":
            target_space = "cross_attention"
        elif args.segmentation_model == "RawValueSpace":
            target_space = "value"
        elif args.segmentation_model == "RawOutputSpace":
            target_space = "output"

    elif args.segmentation_model == "CLIPTextSpan":
        segmentation_model = CLIPTextSpanSegmentationModel(
            device=args.device
        )
    elif args.segmentation_model == "DAAMSDXL":
        segmentation_model = DAAMStableDiffusionXLSegmentationModel(
            device=args.device
        )
    else:
        raise ValueError(f"Segmentation model {args.segmentation_model} not recognized.")

    # # Make transforms for the images and labels (masks)
    image_transforms = transforms.Compose([
        transforms.Resize((1024, 1024), Image.NEAREST),
        # transforms.ToTensor(),
    ])
    label_transforms = transforms.Compose([
    #     # transforms.Resize((224, 224), Image.NEAREST),
        transforms.Resize((64, 64), Image.NEAREST),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255),
    ])

    # dataset = ImagenetSegmentation()
    dataset = VOCSegmentation(
        root="data",
        image_set="val",
        download=False,
        transform=image_transforms,
        target_transform=label_transforms
    )
    # Iterate through the first N images, run concept encoding
    total_correct = 0.0
    total_label = 0.0
    total_intersection_per_class = np.zeros(21)
    total_union_per_class = np.zeros(21)

    total_ap = []
    for index in range(len(dataset)):
        img, labels, present_classes = dataset[index]
        # Remove backgorund from present classes
        present_classes = [class_name for class_name in present_classes if class_name != "background"]
        # Apply transformations
        # img = image_transforms(img)
        # labels = label_transforms(labels)
        if args.segmentation_model == "RawCrossAttention" or args.segmentation_model == "RawOutputSpace" or args.segmentation_model == "RawValueSpace":
            # Run the segmentation model
            predicted_concepts, coefficients, reconstructed_image = segmentation_model(
                img,
                background_concepts=args.background_concepts,
                target_concepts=present_classes,
                caption=",".join([f"a {class_name}" for class_name in present_classes]),
                # concept_scale_values=[1.0] * len(present_classes), TODO Implement this
                stop_after_multimodal_attentions=True,
                offload=False,
                num_samples=args.num_samples,
                device=args.device,
                num_steps=args.num_steps,
                noise_timestep=args.noise_timestep,
                normalize_concepts=args.normalize_concepts,
                # softmax=args.softmax,
                layers=args.layers,
                target_space=target_space,
                joint_attention_kwargs={
                    "concept_cross_attention": args.concept_cross_attention,
                    "concept_self_attention": args.concept_self_attention
                }
            )
        else:
            _, coefficients, reconstructed_image = segmentation_model(
                [img],
                None,
                args.background_concepts + present_classes, # Must be background then present classes
                captions=[",".join([f"a {class_name}" for class_name in present_classes])],
            )
            # Take the argmax over the coefficients
            predicted_concepts = coefficients[0].argmax(0)
            coefficients = coefficients[0]

            if isinstance(coefficients, torch.Tensor):
                coefficients = coefficients.cpu().numpy()
        # Apply softmax to the coefficients
        # coefficients = torch.nn.functional.softmax(coefficients, dim=0)
        # Map the predictions to the appropriate VOC classes
        mask = map_predictions_to_voc_class_indices(
            predicted_concepts,
            present_classes,
            background_concepts=args.background_concepts,
        )
        # Upscale the mask to 224 x 224
        mask = torch.nn.functional.interpolate(
            mask.unsqueeze(0).unsqueeze(0).float(),
            # size=(224, 224),
            size=(64, 64),
            mode="nearest"
        ).squeeze()
        mask = mask.detach().cpu().numpy()
        reconstructed_image = reconstructed_image[0] if isinstance(reconstructed_image, list) else reconstructed_image
        # labels = labels.bool().detach().cpu().numpy().squeeze()
        mask = torch.Tensor(mask)
        # unpadded_mask = torch.stack((1 - mask, mask))
        # unpadded_target = torch.stack((1 - unpadded_target, unpadded_target))
        current_correct, current_label = batch_pix_accuracy(mask, labels) # (batch_size, h * w)
        this_image_acc = current_correct / (current_label + 1e-6)
        total_correct += current_correct
        total_label += current_label
        # Add a 1 - mask and 1 - label
        this_image_mIoU = 0.0
        num_nonzero_classes = 0
        for class_index in range(21):
            intersection = (mask == class_index) & (labels == class_index)
            union = (mask == class_index) | (labels == class_index)
            total_intersection_per_class[class_index] += intersection.sum()
            total_union_per_class[class_index] += union.sum()
            if union.sum() == 0:
                continue
            num_nonzero_classes += 1
            this_image_mIoU += intersection.sum() / union.sum()
        this_image_mIoU /= (num_nonzero_classes + 1e-6)

        mIoU = 0.0
        num_nonzero_classes = 0
        for class_index in range(21):
            if total_union_per_class[class_index] == 0:
                continue
            num_nonzero_classes += 1
            mIoU += total_intersection_per_class[class_index] / total_union_per_class[class_index]

        mIoU /= (num_nonzero_classes + 1e-6)
        # current_inter, current_union = batch_intersection_union(mask, labels, nclass=21)
        # total_inter += current_inter
        # total_union += current_union
        # unpadded_coefficients = torch.stack((1 - unpadded_coefficients, unpadded_coefficients)).unsqueeze(0)
        # labels = torch.Tensor(labels).unsqueeze(0)
        # ap_score = np.nan_to_num(
        #     get_ap_scores(unpadded_coefficients, labels)
        # )
        # total_ap += [ap_score]
        pixAcc = (
            np.float64(1.0)
            * total_correct
            / (np.spacing(1, dtype=np.float64) + total_label)
        )
        # IoU = (
        #     np.float64(1.0)
        #     * total_inter
        #     / (np.spacing(1, dtype=np.float64) + total_union)
        # )
        # mIoU = IoU.mean()
        mAp = np.mean(total_ap)
        print(f"Current pixelwise accuracy: {pixAcc: .4f}, Current average IoU: {mIoU: .4f}, Current mAP: {mAp :.4f}")
        # Plot the results
        fig, axs = plt.subplots(2, len(present_classes) + 1, figsize=(4 * (len(present_classes) + 1), 8))
        plt.suptitle(f"ImageNet Segmentation Results: mIoU: {this_image_mIoU:.4f}, Acc: {this_image_acc:.4f}")
        # Plot the image
        axs[0, 0].imshow(img)
        axs[0, 0].axis("off")
        axs[0, 0].set_title("Image")
        # Plot each of the concept coefficients
        # Sum the backgroudn concepts
        background_coefficients = coefficients[:len(args.background_concepts)].sum(0)
        axs[0, 1].imshow(background_coefficients)
        axs[0, 1].axis("off")
        axs[0, 1].set_title("Background")
        for i, present_class in enumerate(present_classes):
            class_coefficients = coefficients[len(args.background_concepts) + i]
            axs[0, i + 1].imshow(class_coefficients)
            axs[0, i + 1].set_title(present_class)
            axs[0, i + 1].axis("off")
            # Plot the ground truth mask
            class_index = VOCSegmentation.CLASSES_NAMES.index(present_class)
            axs[1, i + 1].imshow(labels.squeeze() == class_index)
            axs[1, i + 1].axis("off")

        plt.savefig(f"{args.image_save_dir}/imagenet_segmentation_{index}.png", dpi=300)
        plt.close()