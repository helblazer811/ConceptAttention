"""
    This is just a wrapper around the various baselines implemented in the 
    Chefer et. al. Transformer Explainability repository. 

    Implements
    - CheferLRPSegmentationModel
    - CheferRolloutSegmentationModel
    - CheferLastLayerAttentionSegmentationModel
    - CheferAttentionGradCAMSegmentationModel
    - CheferTransformerAttributionSegmentationModel
    - CheferFullLRPSegmentationModel
    - CheferLastLayerLRPSegmentationModel
"""

#  # segmentation test for the rollout baseline
#     if args.method == 'rollout':
#         Res = baselines.generate_rollout(image.cuda(), start_layer=1).reshape(batch_size, 1, 14, 14)
    
#     # segmentation test for the LRP baseline (this is full LRP, not partial)
#     elif args.method == 'full_lrp':
#         Res = orig_lrp.generate_LRP(image.cuda(), method="full").reshape(batch_size, 1, 224, 224)
    
#     # segmentation test for our method
#     elif args.method == 'transformer_attribution':
#         Res = lrp.generate_LRP(image.cuda(), start_layer=1, method="transformer_attribution").reshape(batch_size, 1, 14, 14)
    
#     # segmentation test for the partial LRP baseline (last attn layer)
#     elif args.method == 'lrp_last_layer':
#         Res = orig_lrp.generate_LRP(image.cuda(), method="last_layer", is_ablation=args.is_ablation)\
#             .reshape(batch_size, 1, 14, 14)
    
#     # segmentation test for the raw attention baseline (last attn layer)
#     elif args.method == 'attn_last_layer':
#         Res = orig_lrp.generate_LRP(image.cuda(), method="last_layer_attn", is_ablation=args.is_ablation)\
#             .reshape(batch_size, 1, 14, 14)
    
#     # segmentation test for the GradCam baseline (last attn layer)
#     elif args.method == 'attn_gradcam':
#         Res = baselines.generate_cam_attn(image.cuda()).reshape(batch_size, 1, 14, 14)

#     if args.method != 'full_lrp':
#         # interpolate to full image size (224,224)
#         Res = torch.nn.functional.interpolate(Res, scale_factor=16, mode='bilinear').cuda()

import torch
import PIL

from concept_attention.binary_segmentation_baselines.chefer_vit_explainability.ViT_explanation_generator import LRP
from concept_attention.segmentation import SegmentationAbstractClass
from concept_attention.binary_segmentation_baselines.chefer_vit_explainability.ViT_explanation_generator import Baselines, LRP
from concept_attention.binary_segmentation_baselines.chefer_vit_explainability.ViT_new import vit_base_patch16_224
from concept_attention.binary_segmentation_baselines.chefer_vit_explainability.ViT_LRP import vit_base_patch16_224 as vit_LRP
from concept_attention.binary_segmentation_baselines.chefer_vit_explainability.ViT_orig_LRP import vit_base_patch16_224 as vit_orig_LRP


# # Model
# model = vit_base_patch16_224(pretrained=True).cuda()
# baselines = Baselines(model)

# # LRP
# model_LRP = vit_LRP(pretrained=True).cuda()
# model_LRP.eval()
# lrp = LRP(model_LRP)

# # orig LRP
# model_orig_LRP = vit_orig_LRP(pretrained=True).cuda()
# model_orig_LRP.eval()
# orig_lrp = LRP(model_orig_LRP)

# model.eval()

class CheferLRPSegmentationModel(SegmentationAbstractClass):

    def __init__(
        self,
        device: str = "cuda",
        width: int = 224,
        height: int = 224,
    ):
        """
            Initialize the segmentation model.
        """
        super(CheferLRPSegmentationModel, self).__init__()
        self.width = width
        self.height = height
        self.device = device
        # Load the LRP model
        model_orig_LRP = vit_orig_LRP(pretrained=True).to(self.device)
        model_orig_LRP.eval()
        self.orig_lrp = LRP(model_orig_LRP)

    def segment_individual_image(self, image: torch.Tensor, concepts: list[str], caption: str, **kwargs):
        """
            Takes a real image and generates a concept segmentation map
            it by adding noise and running the DiT on it. 
        """
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
            
        prediction_map = self.orig_lrp.generate_LRP(
            image.to(self.device), 
            method="full"
        )
        prediction_map = prediction_map.unsqueeze(0)
        # Rescale the prediction map to 64x64
        prediction_map = torch.nn.functional.interpolate(
            prediction_map, 
            size=(self.width, self.height), 
            mode="nearest"
        ).reshape(1, self.width, self.height)

        return prediction_map, None

class CheferRolloutSegmentationModel(SegmentationAbstractClass):

    def __init__(self, device: str = "cuda", width: int = 224, height: int = 224):
        super(CheferRolloutSegmentationModel, self).__init__()
        self.width = width
        self.height = height
        self.device = device
        model = vit_base_patch16_224(pretrained=True).to(device)
        self.baselines = Baselines(model)

    def segment_individual_image(self, image: torch.Tensor, concepts: list[str], caption: str, **kwargs):
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        prediction_map = self.baselines.generate_rollout(
            image.to(self.device), start_layer=1
        ).reshape(1, 1, 14, 14)
        # Rescale the prediction map to 64x64
        prediction_map = torch.nn.functional.interpolate(
            prediction_map, 
            size=(self.width, self.height), 
            mode="nearest"
        ).reshape(1, self.width, self.height)

        return prediction_map, None


class CheferLastLayerAttentionSegmentationModel(SegmentationAbstractClass):

    def __init__(self, device: str = "cuda", width: int = 224, height: int = 224):
        super(CheferLastLayerAttentionSegmentationModel, self).__init__()
        self.width = width
        self.height = height
        self.device = device
        model_orig_LRP = vit_orig_LRP(pretrained=True).to(device)
        model_orig_LRP.eval()
        self.orig_lrp = LRP(model_orig_LRP)

    def segment_individual_image(self, image: torch.Tensor, concepts: list[str], caption: str, **kwargs):
        if len(image.shape) == 3:
            image = image.unsqueeze(0)

        prediction_map = self.orig_lrp.generate_LRP(
            image.to(self.device), method="last_layer_attn"
        ).reshape(1, 1, 14, 14)
        # Rescale the prediction map to 64x64
        prediction_map = torch.nn.functional.interpolate(
            prediction_map, 
            size=(self.width, self.height), 
            mode="nearest"
        ).reshape(1, self.width, self.height)
        
        return prediction_map, None


class CheferAttentionGradCAMSegmentationModel(SegmentationAbstractClass):

    def __init__(self, device: str = "cuda", width: int = 224, height: int = 224):
        super(CheferAttentionGradCAMSegmentationModel, self).__init__()
        self.width = width
        self.height = height
        self.device = device
        model = vit_base_patch16_224(pretrained=True).to(device)
        self.baselines = Baselines(model)

    def segment_individual_image(self, image: torch.Tensor, concepts: list[str], caption: str, **kwargs):
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        prediction_map = self.baselines.generate_cam_attn(
            image.to(self.device)
        ).reshape(1, 1, 14, 14)
        # Rescale the prediction map to 64x64
        prediction_map = torch.nn.functional.interpolate(
            prediction_map, 
            size=(self.width, self.height), 
            mode="nearest"
        ).reshape(1, self.width, self.height)

        return prediction_map, None


class CheferTransformerAttributionSegmentationModel(SegmentationAbstractClass):

    def __init__(self, device: str = "cuda", width: int = 224, height: int = 224):
        super(CheferTransformerAttributionSegmentationModel, self).__init__()
        self.width = width
        self.height = height
        self.device = device
        model_LRP = vit_LRP(pretrained=True).to(device)
        model_LRP.eval()
        self.lrp = LRP(model_LRP)

    def segment_individual_image(self, image: torch.Tensor, concepts: list[str], caption: str, **kwargs):
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        prediction_map = self.lrp.generate_LRP(
            image.to(self.device), start_layer=1, method="transformer_attribution"
        ).reshape(1, 1, 14, 14)
        # Rescale the prediction map to 64x64
        prediction_map = torch.nn.functional.interpolate(
            prediction_map, 
            size=(self.width, self.height), 
            mode="nearest"
        ).reshape(1, self.width, self.height)

        return prediction_map, None


class CheferFullLRPSegmentationModel(SegmentationAbstractClass):

    def __init__(self, device: str = "cuda", width: int = 224, height: int = 224):
        super(CheferFullLRPSegmentationModel, self).__init__()
        self.width = width
        self.height = height
        self.device = device
        model_LRP = vit_LRP(pretrained=True).to(device)
        model_LRP.eval()
        self.lrp = LRP(model_LRP)

    def segment_individual_image(self, image: torch.Tensor, concepts: list[str], caption: str, **kwargs):
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        prediction_map = self.lrp.generate_LRP(
            image.to(self.device), method="full"
        ).reshape(1, 1, 224, 224)
        # Rescale the prediction map to 64x64
        prediction_map = torch.nn.functional.interpolate(
            prediction_map, 
            size=(self.width, self.height), 
            mode="nearest"
        ).reshape(1, self.width, self.height)

        return prediction_map, None


class CheferLastLayerLRPSegmentationModel(SegmentationAbstractClass):

    def __init__(self, device: str = "cuda", width: int = 224, height: int = 224):
        super(CheferLastLayerLRPSegmentationModel, self).__init__()
        self.width = width
        self.height = height
        self.device = device
        model_LRP = vit_LRP(pretrained=True).to(device)
        model_LRP.eval()
        self.lrp = LRP(model_LRP)

    def segment_individual_image(self, image: torch.Tensor, concepts: list[str], caption: str, **kwargs):
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        prediction_map = self.lrp.generate_LRP(
            image.to(self.device), method="last_layer"
        ).reshape(1, 1, 14, 14)
        # Rescale the prediction map to 64x64
        prediction_map = torch.nn.functional.interpolate(
            prediction_map, 
            size=(self.width, self.height), 
            mode="nearest"
        ).reshape(1, self.width, self.height)

        return prediction_map, None 