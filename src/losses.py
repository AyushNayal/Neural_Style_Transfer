import torch
import torch.nn as nn
import torch.nn.functional as F


def gram_matrix(input):
    """
    Compute the Gram matrix of a feature map.
    The Gram matrix captures style by measuring feature correlations.
    
    input shape: (batch, channels, height, width)
    output shape: (channels, channels)
    """
    b, c, h, w = input.size()
    features = input.view(c, h * w)       # flatten spatial dimensions
    G = torch.mm(features, features.t())  # matrix multiplication to get correlations
    return G.div(c * h * w)               # normalize by total number of elements


class ContentLoss(nn.Module):
    """
    Measures how much the content of the generated image
    differs from the target content image at a specific VGG layer.
    Uses MSE between feature maps.
    """
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()  # freeze — not part of computation graph
        self.loss = 0

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input  # pass input through unchanged (this is a loss probe, not a transform)


class StyleLoss(nn.Module):
    """
    Measures how much the style of the generated image
    differs from the target style image at a specific VGG layer.
    Uses MSE between Gram matrices of feature maps.
    Each layer can have an individual weight.
    """
    def __init__(self, target_feature, weight=1.0):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()
        self.weight = weight
        self.loss = 0

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = self.weight * F.mse_loss(G, self.target)
        return input  # pass through unchanged


class TotalVariationLoss(nn.Module):
    """
    Encourages spatial smoothness in the generated image.
    Penalizes large differences between neighboring pixels.
    Helps reduce high frequency noise and artifacts.
    """
    def __init__(self, weight=1.0):
        super(TotalVariationLoss, self).__init__()
        self.weight = weight
        self.loss = 0

    def forward(self, input):
        self.loss = self.weight * (
            torch.sum(torch.abs(input[:, :, :, :-1] - input[:, :, :, 1:])) +  # horizontal
            torch.sum(torch.abs(input[:, :, :-1, :] - input[:, :, 1:, :]))    # vertical
        )
        return input  # pass through unchanged