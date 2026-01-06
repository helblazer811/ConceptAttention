"""
    Here we evaluate the performance of various models on PascalVOC using only the examples
    that have a single class. 

    We compare our method, and a variety of CLIP based methods. 
"""

import torch
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
import os
import nltk

from concept_attention.binary_segmentation_baselines.clip_text_span_baseline import CLIPTextSpanSegmentationModel
from concept_attention.binary_segmentation_baselines.daam_sd2 import DAAMStableDiffusion2SegmentationModel
from concept_attention.binary_segmentation_baselines.daam_sdxl import DAAMStableDiffusionXLSegmentationModel
from concept_attention.binary_segmentation_baselines.dino import DINOSegmentationModel
nltk.download('wordnet')
from PIL import Image
import argparse


from concept_attention.utils import batch_intersection_union, batch_pix_accuracy, get_ap_scores
from concept_attention.flux.image_generator import FluxGenerator
from concept_attention.binary_segmentation_baselines.chefer_vit_explainability.data.VOC import VOCSegmentation

from new_paper_experiments.pascal_voc_segmentation.multi_class_segmentation import FluxMultiClassSegmentation
from concept_attention.binary_segmentation_baselines.chefer_clip_vit_baselines import CheferAttentionGradCAMSegmentationModel, \
    CheferFullLRPSegmentationModel, CheferLRPSegmentationModel, CheferLastLayerAttentionSegmentationModel, \
    CheferLastLayerLRPSegmentationModel, CheferRolloutSegmentationModel, \
    CheferTransformerAttributionSegmentationModel
from concept_attention.binary_segmentation_baselines.raw_output_space import RawOutputSpaceSegmentationModel
from concept_attention.binary_segmentation_baselines.raw_value_space import RawValueSpaceSegmentationModel
from concept_attention.binary_segmentation_baselines.raw_cross_attention import RawCrossAttentionSegmentationModel


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
        # default="RawOutputSpace",
        default="CheferAttentionGradCAM",
        choices=[
            "RawOutputSpace", 
            "RawValueSpace", 
            "RawCrossAttention",
            "DAAMSDXL",
            "DAAMSD2",
            "CheferLRP",
            "CheferRollout",
            "CheferLastLayerAttention",
            "CheferAttentionGradCAM",
            "CheferTransformerAttribution",
            "CheferFullLRP",
            "CheferLastLayerLRP",
            "CLIPTextSpan",
            "DINO",
        ]
    )
    parser.add_argument("--device", type=str, default="cuda:2")
    parser.add_argument("--normalize_concepts", default=True)
    parser.add_argument("--softmax", action="store_true", default=False)
    parser.add_argument("--offload", default=False)
    # parser.add_argument("--concept_cross_attention", action="store_true")
    # parser.add_argument("--concept_self_attention", action="store_true", default=True)
    parser.add_argument("--downscale_for_eval", action="store_true")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--model_name", type=str, default="flux-schnell")
    parser.add_argument("--num_steps", type=int, default=4)
    parser.add_argument("--apply_blur", action="store_true")
    parser.add_argument("--noise_timestep", type=int, default=3)
    parser.add_argument("--layers", type=int, nargs="+", default=list(range(14, 19)))
    parser.add_argument("--background_concepts", type=list, default=["background", "floor", "grass", "tree", "sky"]) #, "floor", "tree", "person", "grass", "face"])
    # parser.add_argument("--resize_ablation", action="store_true")
    parser.add_argument("--image_save_dir", type=str, default="results/chefer_trans_attribution/segmentations")

    args = parser.parse_args()

    if not os.path.exists(args.image_save_dir):
        os.makedirs(args.image_save_dir)


    # Load up the model
    target_space = None
    if args.segmentation_model == "RawCrossAttention" or \
        args.segmentation_model == "RawOutputSpace" or \
        args.segmentation_model == "RawValueSpace":
        # Load up the model
        generator = FluxGenerator(
            model_name=args.model_name,
            offload=args.offload,
            device=args.device,
        )
        segmentation_model = FluxMultiClassSegmentation(
            generator=generator,
        )
        if args.segmentation_model == "RawOutputSpace":
            target_space = "output"
        elif args.segmentation_model == "RawValueSpace":
            target_space = "value"
        elif args.segmentation_model == "RawCrossAttention":
            target_space = "cross_attention"

    elif args.segmentation_model == "CheferLRP":
        segmentation_model = CheferLRPSegmentationModel(device=args.device)
    elif args.segmentation_model == "CheferRollout":
        segmentation_model = CheferRolloutSegmentationModel(device=args.device)
    elif args.segmentation_model == "CheferLastLayerAttention":
        segmentation_model = CheferLastLayerAttentionSegmentationModel(device=args.device)
    elif args.segmentation_model == "CheferAttentionGradCAM":
        segmentation_model = CheferAttentionGradCAMSegmentationModel(device=args.device)
    elif args.segmentation_model == "CheferTransformerAttribution":
        segmentation_model = CheferTransformerAttributionSegmentationModel(device=args.device)
    elif args.segmentation_model == "CheferFullLRP":
        segmentation_model = CheferFullLRPSegmentationModel(device=args.device)
    elif args.segmentation_model == "CheferLastLayerLRP":
        segmentation_model = CheferLastLayerLRPSegmentationModel(device=args.device)
    elif args.segmentation_model == "DAAMSDXL":
        segmentation_model = DAAMStableDiffusionXLSegmentationModel(device=args.device)
    elif args.segmentation_model == "DAAMSD2":
        segmentation_model = DAAMStableDiffusion2SegmentationModel(device=args.device)
    elif args.segmentation_model == "CLIPTextSpan":
        segmentation_model = CLIPTextSpanSegmentationModel(device=args.device)
    elif args.segmentation_model == "DINO":
        segmentation_model = DINOSegmentationModel(device=args.device)
    else:
        raise ValueError(f"Segmentation model {args.segmentation_model} not recognized.")

    # # Make transforms for the images and labels (masks)
    image_transforms = transforms.Compose([
        # transforms.Resize((224, 224), Image.NEAREST),
        transforms.Resize((512, 512), Image.NEAREST),
        # transforms.ToTensor(),
    ])
    label_transforms = transforms.Compose([
    #     # transforms.Resize((224, 224), Image.NEAREST),
        transforms.Resize((224, 224), Image.NEAREST),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255),
    ])

    # dataset = ImagenetSegmentation()
    dataset = VOCSegmentation(
        root="data",
        image_set="val",
        download=False,
        transform=image_transforms,
        target_transform=label_transforms,
        binary_class=True
    )
    # Iterate through the first N images, run concept encoding
    total_correct = 0.0
    total_label = 0.0
    total_inter = 0.0
    total_union = 0.0
    num_single_class_images = 0

    total_ap = []
    for index in range(len(dataset)):
        img, labels, present_classes = dataset[index]
        # Contiue if present_classes is greater than two "background" and the class
        if len(present_classes) > 2:
            continue
        num_single_class_images += 1
        print(f"Single class image {num_single_class_images}")
        # Remove backgorund from present classes
        present_classes = [class_name for class_name in present_classes if class_name != "background"]
        # Apply transformations
        # img = image_transforms(img)
        # labels = label_transforms(labels)
        if args.segmentation_model == "RawCrossAttention" or \
            args.segmentation_model == "RawOutputSpace" or \
            args.segmentation_model == "RawValueSpace":
            # Run the segmentation model
            predicted_concepts, coefficients, reconstructed_image = segmentation_model(
                img,
                caption=",".join([f"a {class_name}" for class_name in present_classes]),
                background_concepts=args.background_concepts,
                target_concepts=present_classes,
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
                joint_attention_kwargs=None, 
                apply_blur=args.apply_blur,
            )
            mask = map_predictions_to_voc_class_indices(
                predicted_concepts,
                present_classes,
                background_concepts=args.background_concepts,
            )
            # Now merge all non-background classes into 1 and all background to 0, and leave -1
            mask[mask > 0] = 1
            # Also merge the coefficients for the background classes into one
            # assert coefficients.shape[0] == len(args.background_concepts) + 1
            # background_coefficients = coefficients[:len(args.background_concepts)].mean(0)
            # softmaxed_coeffs = torch.softmax(coefficients, dim=0)
            coefficients = coefficients[-1]
            # coefficients = torch.cat((background_coefficients.unsqueeze(0), target_concept.unsqueeze(0)))
            # coefficients = torch.softmax(coefficients, dim=0)
            # coefficients = coefficients[1]
        else:
            # Process input image to tensor
            img = image_transforms(img)
            img_tensor = transforms.ToTensor()(img).to(args.device)
            predicted_concepts, coefficients, reconstructed_image = segmentation_model(
                [img_tensor],
                captions=[f"a {present_classes[0]}"],
                concepts=[present_classes[0]], # These baselines only take a single concept
                target_concepts=[present_classes[0]],
                device=args.device,
                num_samples=args.num_samples,
            )
            mask = torch.Tensor(predicted_concepts[0])
            coefficients = torch.Tensor(coefficients[0])

        # Resize the coefficients to 224 x 224
        coefficients = torch.nn.functional.interpolate(
            coefficients.unsqueeze(0).unsqueeze(0).float(),
            size=(224, 224),
            mode="nearest"
        ).squeeze()
        # Resize the mask to the original image size
        mask = torch.nn.functional.interpolate(
            mask.unsqueeze(0).unsqueeze(0).float(),
            size=(224, 224),
            # size=(64, 64),
            mode="nearest"
        ).squeeze()
        mask = mask.detach().cpu().numpy()
        reconstructed_image = reconstructed_image[0] if isinstance(reconstructed_image, list) else reconstructed_image
        # labels = labels.bool().detach().cpu().numpy().squeeze()
        mask = torch.Tensor(mask)
        labels = torch.Tensor(labels).unsqueeze(0)
        unpadded_mask = torch.stack((1 - mask, mask))
        unpadded_target = torch.stack((1 - labels.squeeze(), labels.squeeze()))
        current_correct, current_label = batch_pix_accuracy(unpadded_mask, unpadded_target) # (batch_size, h * w)
        this_image_acc = current_correct / (current_label + 1e-6)
        total_correct += current_correct
        total_label += current_label
        # IoU
        current_inter, current_union = batch_intersection_union(mask, labels, nclass=2)
        total_inter += current_inter
        total_union += current_union
        # AP score
        unpadded_coefficients = torch.stack((1 - coefficients, coefficients)).unsqueeze(0)
        ap_score = np.nan_to_num(
            get_ap_scores(unpadded_coefficients, labels.squeeze(0))
        )
        total_ap += [ap_score]
        pixAcc = (
            np.float64(1.0)
            * total_correct
            / (np.spacing(1, dtype=np.float64) + total_label)
        )
        IoU = (
            np.float64(1.0)
            * total_inter
            / (np.spacing(1, dtype=np.float64) + total_union)
        )
        mIoU = IoU.mean()
        mAp = np.mean(total_ap)
        print(f"Current pixelwise accuracy: {pixAcc: .4f}, Current average IoU: {mIoU: .4f}, Current mAP: {mAp :.4f}")
        # Plot the results
        fig, axs = plt.subplots(2, 2, figsize=(8, 8))
        plt.suptitle(f"ImageNet Segmentation Results: mIoU: {mIoU:.4f}, Acc: {pixAcc:.4f}")
        # Plot the image
        axs[0, 0].imshow(img)
        axs[0, 0].axis("off")
        axs[0, 0].set_title("Image")
        # Plot each of the concept coefficients
        # Sum the backgroudn concepts
        if args.segmentation_model == "RawCrossAttention" or \
            args.segmentation_model == "RawCrossAttention" or \
            args.segmentation_model == "RawCrossAttention":
            # background_coefficients = coefficients[:len(args.background_concepts)].sum(0)
            # axs[0, 1].imshow(background_coefficients)
            present_class = present_classes[0]
            axs[0, 1].imshow(coefficients)
            axs[0, 1].set_title(present_classes[-1])
            axs[0, 1].axis("off")
            # Plot the ground truth mask
            class_index = VOCSegmentation.CLASSES_NAMES.index(present_class)
            axs[1, 1].imshow(labels.squeeze() == class_index)
            axs[1, 1].axis("off")

            plt.savefig(f"{args.image_save_dir}/imagenet_segmentation_{index}.png", dpi=300)
            plt.close()
        else:
            axs[0, 1].imshow(coefficients)
            axs[0, 1].axis("off")
            # axs[1, 1].imshow(mask)
            # axs[1, 1].axis("off")
            class_index = VOCSegmentation.CLASSES_NAMES.index(present_classes[0])
            axs[1, 1].imshow(labels.squeeze() == class_index)
            axs[1, 1].axis("off")
            plt.savefig(f"{args.image_save_dir}/imagenet_segmentation_{index}.png", dpi=300)
            plt.close()