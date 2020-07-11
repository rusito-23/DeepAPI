"""
Image Utils for the Deep Dream Algorithm.
"""

import torch
from torchvision import transforms as T

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
IMEAN = [- m / s for m, s in zip(MEAN, STD)]
ISTD = [1 / s for s in STD]


def preprocess(image):
    """
    Preprocesses the image to be fed to the Deep Model Algorithm.
    Resizes, normalizes the image and performs Pytorch Tensor Conversion.
    """
    transforms = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transforms(image).unsqueeze(0).requires_grad_(True)


def postprocess(tensor):
    """
    Processes the output for the Deep Dream Algorithm.
    """
    transforms = T.Compose([
        T.Normalize(IMEAN, ISTD),
        T.ToPILImage()
    ])
    return transforms(tensor)


def clip(image):
    """
    Set the image pixel values between 0 and 1.
    """
    for c, (mean, std) in enumerate(zip(MEAN, STD)):
        low, high = -mean / std, (1 - mean) / std
        image[0, c] = torch.clamp(image[0, c], low, high)
    return image
