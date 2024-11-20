import torch
import torch.nn as nn

from ...models.attention_processor import Attention


class WuerstchenLayerNorm(nn.LayerNorm):



class TimestepBlock(nn.Module):



class ResBlock(nn.Module):



# from https://github.com/facebookresearch/ConvNeXt-V2/blob/3608f67cc1dae164790c5d0aead7bf2d73d9719b/models/utils.py#L105
class GlobalResponseNorm(nn.Module):



class AttnBlock(nn.Module):
