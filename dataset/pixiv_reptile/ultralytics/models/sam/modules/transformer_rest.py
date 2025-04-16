# Ultralytics YOLO 🚀, AGPL-3.0 license

import math
from typing import Tuple, Type

import torch
from torch import Tensor, nn

from ultralytics.nn.modules import MLPBlock


class TwoWayTransformer(nn.Module):
    """
    A Two-Way Transformer module that enables the simultaneous attention to both image and query points. This class
    serves as a specialized transformer decoder that attends to an input image using queries whose positional embedding
    is supplied. This is particularly useful for tasks like object detection, image segmentation, and point cloud
    processing.

    Attributes:
        depth (int): The number of layers in the transformer.
        embedding_dim (int): The channel dimension for the input embeddings.
        num_heads (int): The number of heads for multihead attention.
        mlp_dim (int): The internal channel dimension for the MLP block.
        layers (nn.ModuleList): The list of TwoWayAttentionBlock layers that make up the transformer.
        final_attn_token_to_image (Attention): The final attention layer applied from the queries to the image.
        norm_final_attn (nn.LayerNorm): The layer normalization applied to the final queries.
    """




class TwoWayAttentionBlock(nn.Module):
    """
    An attention block that performs both self-attention and cross-attention in two directions: queries to keys and
    keys to queries. This block consists of four main layers: (1) self-attention on sparse inputs, (2) cross-attention
    of sparse inputs to dense inputs, (3) an MLP block on sparse inputs, and (4) cross-attention of dense inputs to
    sparse inputs.

    Attributes:
        self_attn (Attention): The self-attention layer for the queries.
        norm1 (nn.LayerNorm): Layer normalization following the first attention block.
        cross_attn_token_to_image (Attention): Cross-attention layer from queries to keys.
        norm2 (nn.LayerNorm): Layer normalization following the second attention block.
        mlp (MLPBlock): MLP block that transforms the query embeddings.
        norm3 (nn.LayerNorm): Layer normalization following the MLP block.
        norm4 (nn.LayerNorm): Layer normalization following the third attention block.
        cross_attn_image_to_token (Attention): Cross-attention layer from keys to queries.
        skip_first_layer_pe (bool): Whether to skip the positional encoding in the first layer.
    """




class Attention(nn.Module):
    """An attention layer that allows for downscaling the size of the embedding after projection to queries, keys, and
    values.
    """


    @staticmethod

    @staticmethod
