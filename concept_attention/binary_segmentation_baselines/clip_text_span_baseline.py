import torch
import torch.nn.functional as F
import einops 
from torchvision import transforms
from tqdm import tqdm
import PIL

from concept_attention.binary_segmentation_baselines.clip_text_span.prs_hook import hook_prs_logger
from concept_attention.binary_segmentation_baselines.clip_text_span.utils.factory import create_model_and_transforms, get_tokenizer
from concept_attention.binary_segmentation_baselines.clip_text_span.utils.openai_templates import OPENAI_IMAGENET_TEMPLATES
from concept_attention.segmentation import SegmentationAbstractClass

class CLIPTextSpanSegmentationModel(SegmentationAbstractClass):

    def __init__(
        self, 
        model_name='ViT-H-14',
        pretrained='laion2b_s32b_b79k',
        device='cuda:3'
    ):
        self.device = device
        # Load up the clip model and the tokenizer
        self.clip_model, _, preprocess = create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.clip_model.to(device)
        self.clip_model.eval()

        context_length = self.clip_model.context_length
        vocab_size = self.clip_model.vocab_size
        self.tokenizer = get_tokenizer(model_name)
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        self.prs = hook_prs_logger(self.clip_model, device)

    def generate_clip_vectors_for_concepts(self, concepts: list[str]):
        """
            Produces a set of clip vectors for each concept by averaging a set of 
            templates.  
        """
        autocast = torch.cuda.amp.autocast
        with torch.no_grad(), autocast():
            zeroshot_weights = []
            for classname in tqdm(concepts):
                texts = [template(classname) for template in OPENAI_IMAGENET_TEMPLATES]
                texts = self.tokenizer(texts).to(self.device)  # tokenize
                class_embeddings = self.clip_model.encode_text(texts)
                class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
            zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(self.device)

        return zeroshot_weights

    def segment_individual_image(self, image: torch.Tensor, concepts: list[str], caption: str, **kwargs):
        # Apply transform to image
        if isinstance(image, PIL.Image.Image):
            image = self.image_transform(image)
        else:
            image = transforms.ToPILImage()(image)
            image = self.image_transform(image)
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        image_size = image.shape[-1]
        # Compute CLIP vectors for each text concept
        concept_vectors = self.generate_clip_vectors_for_concepts(concepts)
        concept_vectors = concept_vectors.detach().cpu()
        # Create the encodings for the image
        self.prs.reinit()
        representation = self.clip_model.encode_image(
            image.to(self.device), attn_method="head", normalize=False
        )
        attentions, _ = self.prs.finalize(representation)
        representation = representation.detach().cpu()
        attentions = attentions.detach().cpu()  # [b, l, n, h, d]
        # chosen_class = (representation @ concept_vectors).argmax(axis=1)
        attentions_collapse = attentions[:, :, 1:].sum(axis=(1, 3))
        concept_heatmaps = (
            attentions_collapse @ concept_vectors
        )  # [b, n, classes]
        # Now reshape the heatmaps
        patches = image_size // self.clip_model.visual.patch_size[0]
        concept_heatmaps = einops.rearrange(
            concept_heatmaps, 
            "1 (h w) concepts -> concepts h w",
            h=patches, w=patches
        )
        # NOTE: none corresponds to reconstructed image which does not exist for this model
        return concept_heatmaps, None
