"""
Shared output dataclass for all concept attention pipelines.
"""
from dataclasses import dataclass
from typing import List, Union

import numpy as np
import PIL.Image


@dataclass
class ConceptAttentionOutput:
    """
    Standard output for all concept attention pipelines.

    Attributes:
        image: Generated or processed image(s). Can be:
            - PIL.Image.Image for single images
            - np.ndarray for raw arrays
            - List for video frames
        concept_heatmaps: List of PIL images showing concept attention heatmaps,
            one per concept.
        cross_attention_maps: Optional list of PIL images showing cross-attention
            heatmaps, one per concept. May be None if not computed.
    """
    image: Union[PIL.Image.Image, np.ndarray, List]
    concept_heatmaps: List[PIL.Image.Image]
    cross_attention_maps: List[PIL.Image.Image] = None
