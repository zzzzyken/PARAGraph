"""
PARAGraph Models

This package contains all models of the PARAGraph project:
- LesionUNet: ResNet50/Swin UNet for lesion segmentation
- OpticUNet: ResNet50 UNet for optic-disc segmentation and fovea localization
- PARAGraph: hierarchical fundus lesion-graph builder
"""

from .lesion_unet import LesionUNet
from .hacdr_lesion_unet import HACDRLesionUNet
from .optic_unet import OpticUNet
from .paragraph import PARAGraph
from .swin_classifier import SwinClassifier
from .vit_classifier import ViTClassifier
from .densenet_classifier import DenseNet121Classifier
from .efficientnet_classifier import EfficientNetClassifier
from .cabnet_classifier import CABNetClassifier
from .swin_encoder import SwinEncoder
from .mdgnet import MDGNet
from .pranet import PRANet, PRANetLesionDetector

__all__ = [
    'LesionUNet',
    'HACDRLesionUNet',
    'OpticUNet',
    'PARAGraph',
    'SwinClassifier',
    'ViTClassifier',
    'DenseNet121Classifier',
    'EfficientNetClassifier',
    'CABNetClassifier',
    'SwinEncoder',
    'MDGNet',
    'PRANet',
    'PRANetLesionDetector',
]
