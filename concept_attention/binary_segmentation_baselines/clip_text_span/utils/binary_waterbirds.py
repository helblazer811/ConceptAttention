import os
import os.path
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
from typing import Union

from PIL import Image
import pandas as pd
from torchvision.datasets import VisionDataset
import torch


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")

class BinaryWaterbirds(VisionDataset):
    def __init__(
        self,
        root: str,
        split: str,
        loader: Callable[[str], Any] = pil_loader,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
    
        self.loader = loader
        csv = pd.read_csv(os.path.join(root, 'metadata.csv'))
        split = {'test': 2, 'valid': 1, 'train': 0}[split]
        csv = csv[csv['split'] == split]
        self.samples = [(os.path.join(root, csv.iloc[i]['img_filename']), csv.iloc[i]['y']) for i in range(len(csv))]
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self) -> int:
        return len(self.samples)
