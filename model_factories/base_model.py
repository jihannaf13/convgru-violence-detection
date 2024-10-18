import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights, resnet50, ResNet50_Weights, mobilenet_v2

class MobileNetV3Extractor(nn.Module):
    """
    MobileNetV3 feature extractor.

    This module extracts features from input using MobileNetV3 Small architecture.

    Attributes:
    - features (torch.nn.Sequential): Feature extraction layers from MobileNetV3 Small.
    
    Methods:
    - __init__: Initializes MobileNetV3Extractor with MobileNetV3 Small architecture.
    - forward: Performs forward pass to extract features from input.

    """
    def __init__(self):
        """
        Initializes MobileNetV3Extractor with MobileNetV3 Small architecture.
        """
        super(MobileNetV3Extractor, self).__init__()
        mobilenet = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        self.features = nn.Sequential(*list(mobilenet.children())[:-1])
        

    def forward(self, x):
        """
        Performs forward pass to extract features from input.

        Args:
        - x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
        - features (torch.Tensor): Extracted features tensor.
        """
        x = self.features(x)
        return x
    
class MobileNetV1Extractor(nn.Module):
    def __init__(self):
        """
        Initializes MobileNetV1Extractor.
        """
        super(MobileNetV1Extractor, self).__init__()
        mobilenet = mobilenet_v2(pretrained = True)
        self.features = nn.Sequential(*list(mobilenet.children())[:-1])
        

    def forward(self, x):
        """
        Performs forward pass to extract features from input.

        Args:
        - x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
        - features (torch.Tensor): Extracted features tensor.
        """
        x = self.features(x)
        return x
    
class ResNet50Extractor(nn.Module):
    """
    ResNet50 feature extractor.

    This module extracts features from input using ResNet50 Small architecture.

    Attributes:
    - features (torch.nn.Sequential): Feature extraction layers from ResNet50 Small.
    
    Methods:
    - __init__: Initializes ResNet50Extractor with ResNet50 Small architecture.
    - forward: Performs forward pass to extract features from input.

    """
    def __init__(self):
        """
        Initializes ResNet50Extractor with ResNet50 Small architecture.
        """
        super(ResNet50Extractor, self).__init__()
        resnet_50 = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.features = nn.Sequential(*list(resnet_50.children())[:-2])
        

    def forward(self, x):
        """
        Performs forward pass to extract features from input.

        Args:
        - x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
        - features (torch.Tensor): Extracted features tensor.
        """
        x = self.features(x)
        return x