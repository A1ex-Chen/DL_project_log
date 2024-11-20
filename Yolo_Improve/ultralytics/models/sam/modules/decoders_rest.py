# Ultralytics YOLO 🚀, AGPL-3.0 license

from typing import List, Tuple, Type

import torch
from torch import nn
from torch.nn import functional as F

from ultralytics.nn.modules import LayerNorm2d


class MaskDecoder(nn.Module):
    """
    Decoder module for generating masks and their associated quality scores, using a transformer architecture to predict
    masks given image and prompt embeddings.

    Attributes:
        transformer_dim (int): Channel dimension for the transformer module.
        transformer (nn.Module): The transformer module used for mask prediction.
        num_multimask_outputs (int): Number of masks to predict for disambiguating masks.
        iou_token (nn.Embedding): Embedding for the IoU token.
        num_mask_tokens (int): Number of mask tokens.
        mask_tokens (nn.Embedding): Embedding for the mask tokens.
        output_upscaling (nn.Sequential): Neural network sequence for upscaling the output.
        output_hypernetworks_mlps (nn.ModuleList): Hypernetwork MLPs for generating masks.
        iou_prediction_head (nn.Module): MLP for predicting mask quality.
    """





class MLP(nn.Module):
    """
    MLP (Multi-Layer Perceptron) model lightly adapted from
    https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py
    """

