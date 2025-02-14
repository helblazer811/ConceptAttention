import torch
from torchvision import transforms
import torch.nn as nn
import numpy as np

from concept_attention.segmentation import SegmentationAbstractClass
import concept_attention.binary_segmentation_baselines.dino_src.vision_transformer as vits

class DINOSegmentationModel(SegmentationAbstractClass):
    
    def __init__(self, arch="vit_small", patch_size=8, image_size=480, image_path=None, device="cuda"):
        self.device = device
        # build model
        self.image_size = image_size
        self.patch_size = patch_size
        self.model = vits.__dict__[arch](patch_size=patch_size, num_classes=0)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
        self.model.to(device)
        # Load up the model
        if arch == "vit_small" and patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif arch == "vit_small" and patch_size == 8:
            url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"  # model used for visualizations in our paper
        elif arch == "vit_base" and patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif arch == "vit_base" and patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"

        if url is not None:
            print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
            state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
            self.model.load_state_dict(state_dict, strict=True)
        else:
            print("There is no reference weights available for this model => We use random weights.")

        # Transforms
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def segment_individual_image(self, image, concepts, caption, **kwargs):
        # NOTE: Do nothing with concepts or caption, as this is not a text conditioned approach. 
        if isinstance(image, torch.Tensor):
            image = transforms.Resize(self.image_size)(image)
        else:
            image = self.transform(image)
        # Predict the raw scores. 
        # make the image divisible by the patch size
        w, h = image.shape[1] - image.shape[1] % self.patch_size, image.shape[2] - image.shape[2] % self.patch_size
        image = image[:, :w, :h].unsqueeze(0)

        w_featmap = image.shape[-2] // self.patch_size
        h_featmap = image.shape[-1] // self.patch_size

        attentions = self.model.get_last_selfattention(image.to(self.device))
        nh = attentions.shape[1] # number of head

        # we keep only the output patch attention
        attentions = attentions[0, :, 0, 1:].reshape(nh, -1)
        attentions = attentions.reshape(nh, w_featmap, h_featmap)
        attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=self.patch_size, mode="nearest")[0]
        attentions = torch.mean(attentions, dim=0, keepdim=True)
        attentions = attentions.repeat(len(concepts), 1, 1)

        return attentions, None