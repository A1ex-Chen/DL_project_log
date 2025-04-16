# Ultralytics YOLO 🚀, AGPL-3.0 license

# --------------------------------------------------------
# TinyViT Model Architecture
# Copyright (c) 2022 Microsoft
# Adapted from LeViT and Swin Transformer
#   LeViT: (https://github.com/facebookresearch/levit)
#   Swin: (https://github.com/microsoft/swin-transformer)
# Build the TinyViT Model
# --------------------------------------------------------

import itertools
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from ultralytics.utils.instance import to_2tuple


class Conv2d_BN(torch.nn.Sequential):
    """A sequential container that performs 2D convolution followed by batch normalization."""



class PatchEmbed(nn.Module):
    """Embeds images into patches and projects them into a specified embedding dimension."""




class MBConv(nn.Module):
    """Mobile Inverted Bottleneck Conv (MBConv) layer, part of the EfficientNet architecture."""




class PatchMerging(nn.Module):
    """Merges neighboring patches in the feature map and projects to a new dimension."""




class ConvLayer(nn.Module):
    """
    Convolutional Layer featuring multiple MobileNetV3-style inverted bottleneck convolutions (MBConv).

    Optionally applies downsample operations to the output, and provides support for gradient checkpointing.
    """




class Mlp(nn.Module):
    """
    Multi-layer Perceptron (MLP) for transformer architectures.

    This layer takes an input with in_features, applies layer normalization and two fully-connected layers.
    """




class Attention(torch.nn.Module):
    """
    Multi-head attention module with support for spatial awareness, applying attention biases based on spatial
    resolution. Implements trainable attention biases for each unique offset between spatial positions in the resolution
    grid.

    Attributes:
        ab (Tensor, optional): Cached attention biases for inference, deleted during training.
    """


    @torch.no_grad()



class TinyViTBlock(nn.Module):
    """TinyViT Block that applies self-attention and a local convolution to the input."""





class BasicLayer(nn.Module):
    """A basic TinyViT layer for one stage in a TinyViT architecture."""





class LayerNorm2d(nn.Module):
    """A PyTorch implementation of Layer Normalization in 2D."""




class TinyViT(nn.Module):
    """
    The TinyViT architecture for vision tasks.

    Attributes:
        img_size (int): Input image size.
        in_chans (int): Number of input channels.
        num_classes (int): Number of classification classes.
        embed_dims (List[int]): List of embedding dimensions for each layer.
        depths (List[int]): List of depths for each layer.
        num_heads (List[int]): List of number of attention heads for each layer.
        window_sizes (List[int]): List of window sizes for each layer.
        mlp_ratio (float): Ratio of MLP hidden dimension to embedding dimension.
        drop_rate (float): Dropout rate for drop layers.
        drop_path_rate (float): Drop path rate for stochastic depth.
        use_checkpoint (bool): Use checkpointing for efficient memory usage.
        mbconv_expand_ratio (float): Expansion ratio for MBConv layer.
        local_conv_size (int): Local convolution kernel size.
        layer_lr_decay (float): Layer-wise learning rate decay.

    Note:
        This implementation is generalized to accept a list of depths, attention heads,
        embedding dimensions and window sizes, which allows you to create a
        "stack" of TinyViT models of varying configurations.
    """




    @torch.jit.ignore



        self.patch_embed.apply(lambda x: _set_lr_scale(x, lr_scales[0]))
        i = 0
        for layer in self.layers:
            for block in layer.blocks:
                block.apply(lambda x: _set_lr_scale(x, lr_scales[i]))
                i += 1
            if layer.downsample is not None:
                layer.downsample.apply(lambda x: _set_lr_scale(x, lr_scales[i - 1]))
        assert i == depth
        for m in [self.norm_head, self.head]:
            m.apply(lambda x: _set_lr_scale(x, lr_scales[-1]))

        for k, p in self.named_parameters():
            p.param_name = k


        self.apply(_check_lr_scale)

    def _init_weights(self, m):
        """Initializes weights for linear layers and layer normalization in the given module."""
        if isinstance(m, nn.Linear):
            # NOTE: This initialization is needed only for training.
            # trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        """Returns a dictionary of parameter names where weight decay should not be applied."""
        return {"attention_biases"}

    def forward_features(self, x):
        """Runs the input through the model layers and returns the transformed output."""
        x = self.patch_embed(x)  # x input is (N, C, H, W)

        x = self.layers[0](x)
        start_i = 1

        for i in range(start_i, len(self.layers)):
            layer = self.layers[i]
            x = layer(x)
        batch, _, channel = x.shape
        x = x.view(batch, 64, 64, channel)
        x = x.permute(0, 3, 1, 2)
        return self.neck(x)

    def forward(self, x):
        """Executes a forward pass on the input tensor through the constructed model layers."""
        return self.forward_features(x)