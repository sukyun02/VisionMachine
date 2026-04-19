"""Dataset utilities for CIFAR-100 (timm-free)."""
import os

from torchvision import datasets, transforms

# ImageNet default normalization values
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

# CIFAR-100 fine label -> coarse (superclass) label mapping
# 20 superclasses, each containing 5 fine classes
CIFAR100_FINE_TO_COARSE = [
     4,  1, 14,  8,  0,  6,  7,  7, 18,  3,
     3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
     6, 11,  5, 10,  7,  6, 13, 15,  3, 15,
     0, 11,  1, 10, 12, 14, 16,  9,  5, 11,
     5, 19,  8,  8, 15, 13, 14, 17, 18, 10,
    16,  4, 17,  4,  2,  0, 17,  4, 18, 17,
    10,  3,  2, 12, 12, 16, 12,  1,  9, 19,
     2, 10,  0,  1, 16, 12,  9, 13, 15, 13,
    16, 19,  2,  4,  6, 19,  5,  5,  8, 19,
    18,  1,  2, 15,  6,  0, 17,  8, 14, 13,
]


def build_cifar100_test_transform(input_size=32):
    """Build standard test transform for CIFAR-100."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])


def build_cifar100_dataset(data_path, train=False, transform=None):
    """Build CIFAR-100 dataset."""
    if transform is None:
        transform = build_cifar100_test_transform()
    return datasets.CIFAR100(data_path, train=train, transform=transform, download=True)
