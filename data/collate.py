import torch
import torch.nn as nn

from typing import List, Tuple

import torchvision
import torchvision.transforms as T
from torchvision.transforms.transforms import Compose
from torchvision.transforms.functional import InterpolationMode as IMode

class BaseCollateClass(nn.Module):

    def __init__(self,
                 hr_transform: torchvision.transforms.Compose,
                 lr_transform: torchvision.transforms.Compose):

        super(BaseCollateClass, self).__init__()
        self.hr_transform = hr_transform
        self.lr_transform = lr_transform

    def forward(self, batch):
        """Turns a batch of high resolution images into a transformed tuple of
        high-res, low-res image batches.
            Args:
                batch:
                    A batch of images is provided from the SISRDataset class.
            Returns:
                Images:
                    Two batches of transformed images are returned, one low resolution
                    (lr) and one high resolution (hr).
        """
        batch_size = len(batch)

        hr_transformed = [self.hr_transform(batch[i]).unsqueeze(0) \
                            for i in range(batch_size)]
        lr_transformed = [self.lr_transform(hr_transformed[i][0]).unsqueeze(0) \
                            for i in range(batch_size)]

        transforms = (
            torch.cat(hr_transformed, 0),
            torch.cat(lr_transformed, 0)
        )

        return transforms

class SISRCollateFunction(BaseCollateClass):
    """Implementation of a collate function for images for SISR. Inherits from
    BaseCollateClass to define the transforms concretely.
    Attributes:
        image_size:
            Size of HR image.
        upscale_factor:
            Upscaling factor between HR and LR image.
    """

    def __init__(self,
                 image_size: int = 128,
                 upscale_factor: int = 4,
                 mode: str = "train"):


        lr_transform = T.Compose([
            T.ToPILImage(),
            T.Resize(image_size // upscale_factor, interpolation=IMode.BICUBIC),
            T.ToTensor()
        ])

        if mode == "train":
            hr_transform = T.Compose([
                T.RandomCrop(image_size),
                T.RandomRotation(90),
                T.RandomHorizontalFlip(0.5),
                T.ToTensor()
            ])
        else:
            hr_transform = T.Compose([
                T.CenterCrop(image_size),
                T.ToTensor()
            ])

        super(SISRCollateFunction, self).__init__(hr_transform, lr_transform)