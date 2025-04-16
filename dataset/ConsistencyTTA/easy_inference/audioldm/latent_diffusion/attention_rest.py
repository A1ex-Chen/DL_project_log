from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange

from audioldm.latent_diffusion.util import checkpoint












# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = (
            nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU())
            if not glu
            else GEGLU(dim, inner_dim)
        )

        self.net = nn.Sequential(
            project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)







class FeedForward(nn.Module):



def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(
        num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
    )


class LinearAttention(nn.Module):



class SpatialSelfAttention(nn.Module):



class CrossAttention(nn.Module):
    """
    ### Cross Attention Layer
    This falls-back to self-attention when conditional embeddings are not specified.
    """

    # use_flash_attention: bool = True
    use_flash_attention: bool = False






# class CrossAttention(nn.Module):
# def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
#     super().__init__()
#     inner_dim = dim_head * heads
#     context_dim = default(context_dim, query_dim)

#     self.scale = dim_head ** -0.5
#     self.heads = heads

#     self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
#     self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
#     self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

#     self.to_out = nn.Sequential(
#         nn.Linear(inner_dim, query_dim),
#         nn.Dropout(dropout)
#     )

# def forward(self, x, context=None, mask=None):
#     h = self.heads

#     q = self.to_q(x)
#     context = default(context, x)
#     k = self.to_k(context)
#     v = self.to_v(context)

#     q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

#     sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

#     if exists(mask):
#         mask = rearrange(mask, 'b ... -> b (...)')
#         max_neg_value = -torch.finfo(sim.dtype).max
#         mask = repeat(mask, 'b j -> (b h) () j', h=h)
#         sim.masked_fill_(~mask, max_neg_value)

#     # attention, what we cannot get enough of
#     attn = sim.softmax(dim=-1)

#     out = einsum('b i j, b j d -> b i d', attn, v)
#     out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
#     return self.to_out(out)


class BasicTransformerBlock(nn.Module):




class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """

