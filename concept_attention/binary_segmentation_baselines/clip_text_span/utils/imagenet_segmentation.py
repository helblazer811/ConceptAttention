import os
import torch
import torch.utils.data as data
import numpy as np

from torchvision.datasets import ImageNet

from PIL import Image, ImageFilter
import h5py
from glob import glob


class ImagenetSegmentation(data.Dataset):
    CLASSES = 2

    def __init__(self,
                 path,
                 transform=None,
                 target_transform=None):
        self.path = path
        self.transform = transform
        self.target_transform = target_transform
        self.h5py = None
        tmp = h5py.File(path, 'r')
        self.data_length = len(tmp['/value/img'])
        tmp.close()
        del tmp

    def __getitem__(self, index):

        if self.h5py is None:
            self.h5py = h5py.File(self.path, 'r')

        img = np.array(self.h5py[self.h5py['/value/img'][index, 0]]).transpose((2, 1, 0))
        target = np.array(self.h5py[self.h5py[self.h5py['/value/gt'][index, 0]][0, 0]]).transpose((1, 0))

        img = Image.fromarray(img).convert('RGB')
        target = Image.fromarray(target)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = np.array(self.target_transform(target)).astype('int32')
            target = torch.from_numpy(target).long()

        return img, target

    def __len__(self):
        return self.data_length
