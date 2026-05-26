import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights
from src.losses import ContentLoss, StyleLoss, TotalVariationLoss


# These are the VGG19 layers we use for content and style
# conv_4 captures high level structure for content
# conv_1 to conv_5 capture style at different scales
CONTENT_LAYERS = ['conv_4']
STYLE_LAYERS   = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']


def build_vgg_base(device):
    """
    Load pretrained VGG19 and extract only the features (conv) part.
    We freeze it completely — we never train VGG, only the output image.
    """
    cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()
    return cnn.to(device)


def _build_named_sequential(cnn):
    """
    Rebuild VGG19 features as a named Sequential so we can
    refer to layers by name (conv_1, relu_1, pool_1 etc.)
    Also replaces inplace ReLU with out-of-place to avoid
    autograd issues during optimization.
    """
    model = nn.Sequential()
    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f'conv_{i}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{i}'
            layer = nn.ReLU(inplace=False)  # inplace ReLU breaks gradient flow
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{i}'
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'bn_{i}'
        else:
            raise RuntimeError(f'Unrecognized layer: {layer.__class__.__name__}')

        model.add_module(name, layer)

    return model


def _extract_targets(model, content_img, style_img):
    """
    Do a single forward pass through the named model to
    extract feature map targets for content and style layers.
    These become the fixed targets our loss functions compare against.
    """
    content_targets = {}
    style_targets   = {}
    x = content_img

    for name, layer in model._modules.items():
        x = layer(x)
        if name in CONTENT_LAYERS:
            content_targets[name] = x.detach()

    x = style_img
    for name, layer in model._modules.items():
        x = layer(x)
        if name in STYLE_LAYERS:
            style_targets[name] = x.detach()

    return content_targets, style_targets


def build_model(cnn, content_img, style_img, style_weights, tv_weight, device):
    """
    Build the full model by inserting ContentLoss, StyleLoss, and
    TotalVariationLoss probes at the right points in the VGG19 network.

    Returns:
        final_model   : nn.Sequential with loss probes inserted
        content_losses: list of ContentLoss modules (to read .loss from)
        style_losses  : list of StyleLoss modules (to read .loss from)
        tv_loss       : TotalVariationLoss module
    """
    base_model = _build_named_sequential(cnn)
    content_targets, style_targets = _extract_targets(base_model, content_img, style_img)

    final_model    = nn.Sequential()
    content_losses = []
    style_losses   = []

    i = 0
    for name, layer in base_model._modules.items():
        final_model.add_module(name, layer)

        if name in CONTENT_LAYERS:
            cl = ContentLoss(content_targets[name])
            final_model.add_module(f'content_loss_{i}', cl)
            content_losses.append(cl)

        if name in STYLE_LAYERS:
            weight = style_weights.get(name, 1.0)
            sl = StyleLoss(style_targets[name], weight=weight)
            final_model.add_module(f'style_loss_{i}', sl)
            style_losses.append(sl)

        i += 1

    # TV loss goes at the end — it operates on the full image, not VGG features
    tv = TotalVariationLoss(weight=tv_weight)
    final_model.add_module('tv_loss', tv)

    # Freeze entire model — only the input image gets gradients
    final_model.eval()
    final_model.requires_grad_(False)

    return final_model.to(device), content_losses, style_losses, tv