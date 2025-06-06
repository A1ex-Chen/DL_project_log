# --------------------------------------------------------
# FocalNet for Semantic Segmentation
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Jianwei Yang
# --------------------------------------------------------
import math
import time
import numpy as np
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from detectron2.utils.file_io import PathManager
from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec

from .registry import register_backbone

logger = logging.getLogger(__name__)

class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class FocalModulation(nn.Module):
    """ Focal Modulation

    Args:
        dim (int): Number of input channels.
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        focal_level (int): Number of focal levels
        focal_window (int): Focal window size at focal level 1
        focal_factor (int, default=2): Step to increase the focal window
        use_postln (bool, default=False): Whether use post-modulation layernorm
    """

    def __init__(self, dim, proj_drop=0., focal_level=2, focal_window=7, focal_factor=2, use_postln=False, use_postln_in_modulation=False, scaling_modulator=False):

        super().__init__()
        self.dim = dim

        # specific args for focalv3
        self.focal_level = focal_level
        self.focal_window = focal_window
        self.focal_factor = focal_factor
        self.use_postln_in_modulation = use_postln_in_modulation
        self.scaling_modulator = scaling_modulator

        self.f = nn.Linear(dim, 2*dim+(self.focal_level+1), bias=True)
        self.h = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, groups=1, bias=True)

        self.act = nn.GELU()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.focal_layers = nn.ModuleList()

        if self.use_postln_in_modulation:
            self.ln = nn.LayerNorm(dim)

        for k in range(self.focal_level):
            kernel_size = self.focal_factor*k + self.focal_window
            self.focal_layers.append(
                nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, groups=dim, 
                        padding=kernel_size//2, bias=False),
                    nn.GELU(),
                    )
                )

    def forward(self, x):
        """ Forward function.

        Args:
            x: input features with shape of (B, H, W, C)
        """
        B, nH, nW, C = x.shape
        x = self.f(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        q, ctx, gates = torch.split(x, (C, C, self.focal_level+1), 1)
        
        ctx_all = 0
        for l in range(self.focal_level):                     
            ctx = self.focal_layers[l](ctx)
            ctx_all = ctx_all + ctx*gates[:, l:l+1]
        ctx_global = self.act(ctx.mean(2, keepdim=True).mean(3, keepdim=True))
        ctx_all = ctx_all + ctx_global*gates[:,self.focal_level:]

        if self.scaling_modulator:
            ctx_all = ctx_all / (self.focal_level + 1)

        x_out = q * self.h(ctx_all)
        x_out = x_out.permute(0, 2, 3, 1).contiguous()
        if self.use_postln_in_modulation:
            x_out = self.ln(x_out)            
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)
        return x_out

class FocalModulationBlock(nn.Module):
    """ Focal Modulation Block.

    Args:
        dim (int): Number of input channels.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        focal_level (int): number of focal levels
        focal_window (int): focal kernel size at level 1
    """

    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., 
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 focal_level=2, focal_window=9, 
                 use_postln=False, use_postln_in_modulation=False,
                 scaling_modulator=False, 
                 use_layerscale=False, 
                 layerscale_value=1e-4):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.focal_window = focal_window
        self.focal_level = focal_level
        self.use_postln = use_postln
        self.use_layerscale = use_layerscale

        self.dw1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.modulation = FocalModulation(
            dim, focal_window=self.focal_window, focal_level=self.focal_level, proj_drop=drop, use_postln_in_modulation=use_postln_in_modulation, scaling_modulator=scaling_modulator
        )            

        self.dw2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.H = None
        self.W = None

        self.gamma_1 = 1.0
        self.gamma_2 = 1.0
        if self.use_layerscale:
            self.gamma_1 = nn.Parameter(layerscale_value * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(layerscale_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        x = x + self.dw1(x)
        x = x.permute(0, 2, 3, 1).contiguous().view(B, L, C)

        shortcut = x
        if not self.use_postln:
            x = self.norm1(x)
        x = x.view(B, H, W, C)
        
        # FM
        x = self.modulation(x).view(B, H * W, C)
        x = shortcut + self.drop_path(self.gamma_1 * x)
        if self.use_postln:
            x = self.norm1(x)

        x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        x = x + self.dw2(x)
        x = x.permute(0, 2, 3, 1).contiguous().view(B, L, C)

        if not self.use_postln:
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))        
        else:
            x = x + self.drop_path(self.gamma_2 * self.mlp(x))
            x = self.norm2(x)

        return x

class BasicLayer(nn.Module):
    """ A basic focal modulation layer for one stage.

    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        focal_level (int): Number of focal levels
        focal_window (int): Focal window size at focal level 1
        use_conv_embed (bool): Use overlapped convolution for patch embedding or now. Default: False
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self,
                 dim,
                 depth,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 focal_window=9, 
                 focal_level=2, 
                 use_conv_embed=False,     
                 use_postln=False,          
                 use_postln_in_modulation=False, 
                 scaling_modulator=False,
                 use_layerscale=False,                   
                 use_checkpoint=False, 
                 use_pre_norm=False, 
        ):
        super().__init__()
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            FocalModulationBlock(
                dim=dim,
                mlp_ratio=mlp_ratio,
                drop=drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                focal_window=focal_window, 
                focal_level=focal_level, 
                use_postln=use_postln, 
                use_postln_in_modulation=use_postln_in_modulation, 
                scaling_modulator=scaling_modulator,
                use_layerscale=use_layerscale, 
                norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                patch_size=2,
                in_chans=dim, embed_dim=2*dim, 
                use_conv_embed=use_conv_embed, 
                norm_layer=norm_layer, 
                is_stem=False, 
                use_pre_norm=use_pre_norm
            )

        else:
            self.downsample = None

    def forward(self, x, H, W):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        for blk in self.blocks:
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x_reshaped = x.transpose(1, 2).view(x.shape[0], x.shape[-1], H, W)
            x_down = self.downsample(x_reshaped)   
            x_down = x_down.flatten(2).transpose(1, 2)            
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x, H, W, x_down, Wh, Ww
        else:
            return x, H, W, x, H, W


# class PatchEmbed(nn.Module):
#     r""" Image to Patch Embedding

#     Args:
#         img_size (int): Image size.  Default: 224.
#         patch_size (int): Patch token size. Default: 4.
#         in_chans (int): Number of input image channels. Default: 3.
#         embed_dim (int): Number of linear projection output channels. Default: 96.
#         norm_layer (nn.Module, optional): Normalization layer. Default: None
#     """

#     def __init__(self, img_size=(224, 224), patch_size=4, in_chans=3, embed_dim=96, 
#         use_conv_embed=False, norm_layer=None, is_stem=False, use_pre_norm=False):
#         super().__init__()
#         patch_size = to_2tuple(patch_size)
#         patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.patches_resolution = patches_resolution
#         self.num_patches = patches_resolution[0] * patches_resolution[1]

#         self.in_chans = in_chans
#         self.embed_dim = embed_dim
#         self.use_pre_norm = use_pre_norm

#         if use_conv_embed:
#             # if we choose to use conv embedding, then we treat the stem and non-stem differently
#             if is_stem:
#                 kernel_size = 7; padding = 3; stride = 4
#             else:
#                 kernel_size = 3; padding = 1; stride = 2
#             self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)
#         else:
#             self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
#         if self.use_pre_norm:
#             if norm_layer is not None:
#                 self.norm = norm_layer(in_chans)
#             else:
#                 self.norm = None
#         else:
#             if norm_layer is not None:
#                 self.norm = norm_layer(embed_dim)
#             else:
#                 self.norm = None

#     def forward(self, x):
#         B, C, H, W = x.shape
#         # FIXME look at relaxing size constraints
#         assert H == self.img_size[0] and W == self.img_size[1], \
#             f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        
#         if self.use_pre_norm:
#             if self.norm is not None:
#                 x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
#                 x = self.norm(x).transpose(1, 2).view(B, C, H, W)
#             x = self.proj(x).flatten(2).transpose(1, 2)
#         else:
#             x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
#             if self.norm is not None:
#                 x = self.norm(x)
#         return x

#     def flops(self):
#         Ho, Wo = self.patches_resolution
#         flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
#         if self.norm is not None:
#             flops += Ho * Wo * self.embed_dim
#         return flops

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding

    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
        use_conv_embed (bool): Whether use overlapped convolution for patch embedding. Default: False
        is_stem (bool): Is the stem block or not. 
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, use_conv_embed=False, is_stem=False, use_pre_norm=False):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.use_pre_norm = use_pre_norm

        if use_conv_embed:
            # if we choose to use conv embedding, then we treat the stem and non-stem differently
            if is_stem:
                kernel_size = 7; padding = 3; stride = 4
            else:
                kernel_size = 3; padding = 1; stride = 2
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)                    
        else:
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        if self.use_pre_norm:
            if norm_layer is not None:
                self.norm = norm_layer(in_chans)
            else:
                self.norm = None       
        else:
            if norm_layer is not None:
                self.norm = norm_layer(embed_dim)
            else:
                self.norm = None

    def forward(self, x):
        """Forward function."""
        B, C, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        if self.use_pre_norm:
            if self.norm is not None:
                x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
                x = self.norm(x).transpose(1, 2).view(B, C, H, W)
            x = self.proj(x)
        else:
            x = self.proj(x)  # B C Wh Ww
            if self.norm is not None:
                Wh, Ww = x.size(2), x.size(3)
                x = x.flatten(2).transpose(1, 2)
                x = self.norm(x)
                x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)

        return x


class FocalNet(nn.Module):
    """ FocalNet backbone.

    Args:
        pretrain_img_size (int): Input image size for training the pretrained model,
            used in absolute postion embedding. Default 224.
        patch_size (int | tuple(int)): Patch size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        drop_rate (float): Dropout rate.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True.
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        focal_levels (Sequence[int]): Number of focal levels at four stages
        focal_windows (Sequence[int]): Focal window sizes at first focal level at four stages
        use_conv_embed (bool): Whether use overlapped convolution for patch embedding
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 pretrain_img_size=1600,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 mlp_ratio=4.,
                 drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 out_indices=[0, 1, 2, 3],
                 frozen_stages=-1,
                 focal_levels=[2,2,2,2], 
                 focal_windows=[9,9,9,9],
                 use_pre_norms=[False, False, False, False], 
                 use_conv_embed=False, 
                 use_postln=False, 
                 use_postln_in_modulation=False, 
                 scaling_modulator=False,
                 use_layerscale=False, 
                 use_checkpoint=False, 
        ):
        super().__init__()

        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None, 
            use_conv_embed=use_conv_embed, is_stem=True, use_pre_norm=False)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchEmbed if (i_layer < self.num_layers - 1) else None,
                focal_window=focal_windows[i_layer], 
                focal_level=focal_levels[i_layer], 
                use_pre_norm=use_pre_norms[i_layer], 
                use_conv_embed=use_conv_embed,
                use_postln=use_postln, 
                use_postln_in_modulation=use_postln_in_modulation,
                scaling_modulator=scaling_modulator,
                use_layerscale=use_layerscale, 
                use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features        
        # self.norm = norm_layer(num_features[-1])

        # add a norm layer for each output
        for i_layer in self.out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def load_weights(self, pretrained_dict=None, pretrained_layers=[], verbose=True):
        model_dict = self.state_dict()

        missed_dict = [k for k in model_dict.keys() if k not in pretrained_dict]
        logger.info(f'=> Missed keys {missed_dict}')
        unexpected_dict = [k for k in pretrained_dict.keys() if k not in model_dict]
        logger.info(f'=> Unexpected keys {unexpected_dict}')

        pretrained_dict = {
            k: v for k, v in pretrained_dict.items()
            if k in model_dict.keys()
        }
        
        need_init_state_dict = {}
        for k, v in pretrained_dict.items():
            need_init = (
                (
                    k.split('.')[0] in pretrained_layers
                    or pretrained_layers[0] == '*'
                )
                and 'relative_position_index' not in k
                and 'attn_mask' not in k
            )

            if need_init:
                # if verbose:
                #     logger.info(f'=> init {k} from {pretrained}')

                if ('pool_layers' in k) or ('focal_layers' in k) and v.size() != model_dict[k].size():
                    table_pretrained = v
                    table_current = model_dict[k]
                    fsize1 = table_pretrained.shape[2]
                    fsize2 = table_current.shape[2]

                    # NOTE: different from interpolation used in self-attention, we use padding or clipping for focal conv
                    if fsize1 < fsize2:
                        table_pretrained_resized = torch.zeros(table_current.shape)
                        table_pretrained_resized[:, :, (fsize2-fsize1)//2:-(fsize2-fsize1)//2, (fsize2-fsize1)//2:-(fsize2-fsize1)//2] = table_pretrained
                        v = table_pretrained_resized
                    elif fsize1 > fsize2:
                        table_pretrained_resized = table_pretrained[:, :, (fsize1-fsize2)//2:-(fsize1-fsize2)//2, (fsize1-fsize2)//2:-(fsize1-fsize2)//2]
                        v = table_pretrained_resized


                if ("modulation.f" in k or "pre_conv" in k): 
                    table_pretrained = v
                    table_current = model_dict[k]
                    if table_pretrained.shape != table_current.shape:
                        if len(table_pretrained.shape) == 2:
                            dim = table_pretrained.shape[1]
                            assert table_current.shape[1] == dim
                            L1 = table_pretrained.shape[0]
                            L2 = table_current.shape[0]

                            if L1 < L2:
                                table_pretrained_resized = torch.zeros(table_current.shape)
                                # copy for linear project
                                table_pretrained_resized[:2*dim] = table_pretrained[:2*dim]
                                # copy for global token gating
                                table_pretrained_resized[-1] = table_pretrained[-1]
                                # copy for first multiple focal levels
                                table_pretrained_resized[2*dim:2*dim+(L1-2*dim-1)] = table_pretrained[2*dim:-1]
                                # reassign pretrained weights
                                v = table_pretrained_resized
                            elif L1 > L2:
                                raise NotImplementedError
                        elif len(table_pretrained.shape) == 1:
                            dim = table_pretrained.shape[0]
                            L1 = table_pretrained.shape[0]
                            L2 = table_current.shape[0]
                            if L1 < L2:
                                table_pretrained_resized = torch.zeros(table_current.shape)
                                # copy for linear project
                                table_pretrained_resized[:dim] = table_pretrained[:dim]
                                # copy for global token gating
                                table_pretrained_resized[-1] = table_pretrained[-1]
                                # copy for first multiple focal levels
                                # table_pretrained_resized[dim:2*dim+(L1-2*dim-1)] = table_pretrained[2*dim:-1]
                                # reassign pretrained weights
                                v = table_pretrained_resized
                            elif L1 > L2:
                                raise NotImplementedError    

                need_init_state_dict[k] = v
        
        self.load_state_dict(need_init_state_dict, strict=False)


    def forward(self, x):
        """Forward function."""
        tic = time.time()
        x = self.patch_embed(x)
        Wh, Ww = x.size(2), x.size(3)

        x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        outs = {}
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)

                out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs["res{}".format(i + 2)] = out
                
        if len(self.out_indices) == 0:
            outs["res5"] = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()

        toc = time.time()
        return outs

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(FocalNet, self).train(mode)
        self._freeze_stages()


class D2FocalNet(FocalNet, Backbone):
    def __init__(self, cfg, input_shape):

        pretrain_img_size = cfg['BACKBONE']['FOCAL']['PRETRAIN_IMG_SIZE']
        patch_size = cfg['BACKBONE']['FOCAL']['PATCH_SIZE']
        in_chans = 3
        embed_dim = cfg['BACKBONE']['FOCAL']['EMBED_DIM']
        depths = cfg['BACKBONE']['FOCAL']['DEPTHS']
        mlp_ratio = cfg['BACKBONE']['FOCAL']['MLP_RATIO']
        drop_rate = cfg['BACKBONE']['FOCAL']['DROP_RATE']
        drop_path_rate = cfg['BACKBONE']['FOCAL']['DROP_PATH_RATE']
        norm_layer = nn.LayerNorm
        patch_norm = cfg['BACKBONE']['FOCAL']['PATCH_NORM']
        use_checkpoint = cfg['BACKBONE']['FOCAL']['USE_CHECKPOINT']
        out_indices = cfg['BACKBONE']['FOCAL']['OUT_INDICES']
        scaling_modulator = cfg['BACKBONE']['FOCAL'].get('SCALING_MODULATOR', False)

        super().__init__(
            pretrain_img_size,
            patch_size,
            in_chans,
            embed_dim,
            depths,
            mlp_ratio,
            drop_rate,
            drop_path_rate,
            norm_layer,
            patch_norm,
            out_indices,
            focal_levels=cfg['BACKBONE']['FOCAL']['FOCAL_LEVELS'],
            focal_windows=cfg['BACKBONE']['FOCAL']['FOCAL_WINDOWS'],   
            use_conv_embed=cfg['BACKBONE']['FOCAL']['USE_CONV_EMBED'],    
            use_postln=cfg['BACKBONE']['FOCAL']['USE_POSTLN'],       
            use_postln_in_modulation=cfg['BACKBONE']['FOCAL']['USE_POSTLN_IN_MODULATION'], 
            scaling_modulator=scaling_modulator,
            use_layerscale=cfg['BACKBONE']['FOCAL']['USE_LAYERSCALE'], 
            use_checkpoint=use_checkpoint,
        )

        self._out_features = cfg['BACKBONE']['FOCAL']['OUT_FEATURES']

        self._out_feature_strides = {
            "res2": 4,
            "res3": 8,
            "res4": 16,
            "res5": 32,
        }
        self._out_feature_channels = {
            "res2": self.num_features[0],
            "res3": self.num_features[1],
            "res4": self.num_features[2],
            "res5": self.num_features[3],
        }

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.
        Returns:
            dict[str->Tensor]: names and the corresponding features
        """
        assert (
            x.dim() == 4
        ), f"SwinTransformer takes an input of shape (N, C, H, W). Got {x.shape} instead!"
        outputs = {}
        y = super().forward(x)
        for k in y.keys():
            if k in self._out_features:
                outputs[k] = y[k]
        return outputs

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

    @property
    def size_divisibility(self):
        return 32

@register_backbone


class FocalModulation(nn.Module):
    """ Focal Modulation

    Args:
        dim (int): Number of input channels.
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        focal_level (int): Number of focal levels
        focal_window (int): Focal window size at focal level 1
        focal_factor (int, default=2): Step to increase the focal window
        use_postln (bool, default=False): Whether use post-modulation layernorm
    """



class FocalModulationBlock(nn.Module):
    """ Focal Modulation Block.

    Args:
        dim (int): Number of input channels.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        focal_level (int): number of focal levels
        focal_window (int): focal kernel size at level 1
    """



class BasicLayer(nn.Module):
    """ A basic focal modulation layer for one stage.

    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        focal_level (int): Number of focal levels
        focal_window (int): Focal window size at focal level 1
        use_conv_embed (bool): Use overlapped convolution for patch embedding or now. Default: False
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """




# class PatchEmbed(nn.Module):
#     r""" Image to Patch Embedding

#     Args:
#         img_size (int): Image size.  Default: 224.
#         patch_size (int): Patch token size. Default: 4.
#         in_chans (int): Number of input image channels. Default: 3.
#         embed_dim (int): Number of linear projection output channels. Default: 96.
#         norm_layer (nn.Module, optional): Normalization layer. Default: None
#     """

#     def __init__(self, img_size=(224, 224), patch_size=4, in_chans=3, embed_dim=96, 
#         use_conv_embed=False, norm_layer=None, is_stem=False, use_pre_norm=False):
#         super().__init__()
#         patch_size = to_2tuple(patch_size)
#         patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.patches_resolution = patches_resolution
#         self.num_patches = patches_resolution[0] * patches_resolution[1]

#         self.in_chans = in_chans
#         self.embed_dim = embed_dim
#         self.use_pre_norm = use_pre_norm

#         if use_conv_embed:
#             # if we choose to use conv embedding, then we treat the stem and non-stem differently
#             if is_stem:
#                 kernel_size = 7; padding = 3; stride = 4
#             else:
#                 kernel_size = 3; padding = 1; stride = 2
#             self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)
#         else:
#             self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
#         if self.use_pre_norm:
#             if norm_layer is not None:
#                 self.norm = norm_layer(in_chans)
#             else:
#                 self.norm = None
#         else:
#             if norm_layer is not None:
#                 self.norm = norm_layer(embed_dim)
#             else:
#                 self.norm = None

#     def forward(self, x):
#         B, C, H, W = x.shape
#         # FIXME look at relaxing size constraints
#         assert H == self.img_size[0] and W == self.img_size[1], \
#             f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        
#         if self.use_pre_norm:
#             if self.norm is not None:
#                 x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
#                 x = self.norm(x).transpose(1, 2).view(B, C, H, W)
#             x = self.proj(x).flatten(2).transpose(1, 2)
#         else:
#             x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
#             if self.norm is not None:
#                 x = self.norm(x)
#         return x

#     def flops(self):
#         Ho, Wo = self.patches_resolution
#         flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
#         if self.norm is not None:
#             flops += Ho * Wo * self.embed_dim
#         return flops

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding

    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
        use_conv_embed (bool): Whether use overlapped convolution for patch embedding. Default: False
        is_stem (bool): Is the stem block or not. 
    """




class FocalNet(nn.Module):
    """ FocalNet backbone.

    Args:
        pretrain_img_size (int): Input image size for training the pretrained model,
            used in absolute postion embedding. Default 224.
        patch_size (int | tuple(int)): Patch size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        drop_rate (float): Dropout rate.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True.
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        focal_levels (Sequence[int]): Number of focal levels at four stages
        focal_windows (Sequence[int]): Focal window sizes at first focal level at four stages
        use_conv_embed (bool): Whether use overlapped convolution for patch embedding
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """









class D2FocalNet(FocalNet, Backbone):



    @property

        if isinstance(pretrained, str):
            self.apply(_init_weights)
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def load_weights(self, pretrained_dict=None, pretrained_layers=[], verbose=True):
        model_dict = self.state_dict()

        missed_dict = [k for k in model_dict.keys() if k not in pretrained_dict]
        logger.info(f'=> Missed keys {missed_dict}')
        unexpected_dict = [k for k in pretrained_dict.keys() if k not in model_dict]
        logger.info(f'=> Unexpected keys {unexpected_dict}')

        pretrained_dict = {
            k: v for k, v in pretrained_dict.items()
            if k in model_dict.keys()
        }
        
        need_init_state_dict = {}
        for k, v in pretrained_dict.items():
            need_init = (
                (
                    k.split('.')[0] in pretrained_layers
                    or pretrained_layers[0] == '*'
                )
                and 'relative_position_index' not in k
                and 'attn_mask' not in k
            )

            if need_init:
                # if verbose:
                #     logger.info(f'=> init {k} from {pretrained}')

                if ('pool_layers' in k) or ('focal_layers' in k) and v.size() != model_dict[k].size():
                    table_pretrained = v
                    table_current = model_dict[k]
                    fsize1 = table_pretrained.shape[2]
                    fsize2 = table_current.shape[2]

                    # NOTE: different from interpolation used in self-attention, we use padding or clipping for focal conv
                    if fsize1 < fsize2:
                        table_pretrained_resized = torch.zeros(table_current.shape)
                        table_pretrained_resized[:, :, (fsize2-fsize1)//2:-(fsize2-fsize1)//2, (fsize2-fsize1)//2:-(fsize2-fsize1)//2] = table_pretrained
                        v = table_pretrained_resized
                    elif fsize1 > fsize2:
                        table_pretrained_resized = table_pretrained[:, :, (fsize1-fsize2)//2:-(fsize1-fsize2)//2, (fsize1-fsize2)//2:-(fsize1-fsize2)//2]
                        v = table_pretrained_resized


                if ("modulation.f" in k or "pre_conv" in k): 
                    table_pretrained = v
                    table_current = model_dict[k]
                    if table_pretrained.shape != table_current.shape:
                        if len(table_pretrained.shape) == 2:
                            dim = table_pretrained.shape[1]
                            assert table_current.shape[1] == dim
                            L1 = table_pretrained.shape[0]
                            L2 = table_current.shape[0]

                            if L1 < L2:
                                table_pretrained_resized = torch.zeros(table_current.shape)
                                # copy for linear project
                                table_pretrained_resized[:2*dim] = table_pretrained[:2*dim]
                                # copy for global token gating
                                table_pretrained_resized[-1] = table_pretrained[-1]
                                # copy for first multiple focal levels
                                table_pretrained_resized[2*dim:2*dim+(L1-2*dim-1)] = table_pretrained[2*dim:-1]
                                # reassign pretrained weights
                                v = table_pretrained_resized
                            elif L1 > L2:
                                raise NotImplementedError
                        elif len(table_pretrained.shape) == 1:
                            dim = table_pretrained.shape[0]
                            L1 = table_pretrained.shape[0]
                            L2 = table_current.shape[0]
                            if L1 < L2:
                                table_pretrained_resized = torch.zeros(table_current.shape)
                                # copy for linear project
                                table_pretrained_resized[:dim] = table_pretrained[:dim]
                                # copy for global token gating
                                table_pretrained_resized[-1] = table_pretrained[-1]
                                # copy for first multiple focal levels
                                # table_pretrained_resized[dim:2*dim+(L1-2*dim-1)] = table_pretrained[2*dim:-1]
                                # reassign pretrained weights
                                v = table_pretrained_resized
                            elif L1 > L2:
                                raise NotImplementedError    

                need_init_state_dict[k] = v
        
        self.load_state_dict(need_init_state_dict, strict=False)


    def forward(self, x):
        """Forward function."""
        tic = time.time()
        x = self.patch_embed(x)
        Wh, Ww = x.size(2), x.size(3)

        x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        outs = {}
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)

                out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs["res{}".format(i + 2)] = out
                
        if len(self.out_indices) == 0:
            outs["res5"] = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()

        toc = time.time()
        return outs

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(FocalNet, self).train(mode)
        self._freeze_stages()


class D2FocalNet(FocalNet, Backbone):
    def __init__(self, cfg, input_shape):

        pretrain_img_size = cfg['BACKBONE']['FOCAL']['PRETRAIN_IMG_SIZE']
        patch_size = cfg['BACKBONE']['FOCAL']['PATCH_SIZE']
        in_chans = 3
        embed_dim = cfg['BACKBONE']['FOCAL']['EMBED_DIM']
        depths = cfg['BACKBONE']['FOCAL']['DEPTHS']
        mlp_ratio = cfg['BACKBONE']['FOCAL']['MLP_RATIO']
        drop_rate = cfg['BACKBONE']['FOCAL']['DROP_RATE']
        drop_path_rate = cfg['BACKBONE']['FOCAL']['DROP_PATH_RATE']
        norm_layer = nn.LayerNorm
        patch_norm = cfg['BACKBONE']['FOCAL']['PATCH_NORM']
        use_checkpoint = cfg['BACKBONE']['FOCAL']['USE_CHECKPOINT']
        out_indices = cfg['BACKBONE']['FOCAL']['OUT_INDICES']
        scaling_modulator = cfg['BACKBONE']['FOCAL'].get('SCALING_MODULATOR', False)

        super().__init__(
            pretrain_img_size,
            patch_size,
            in_chans,
            embed_dim,
            depths,
            mlp_ratio,
            drop_rate,
            drop_path_rate,
            norm_layer,
            patch_norm,
            out_indices,
            focal_levels=cfg['BACKBONE']['FOCAL']['FOCAL_LEVELS'],
            focal_windows=cfg['BACKBONE']['FOCAL']['FOCAL_WINDOWS'],   
            use_conv_embed=cfg['BACKBONE']['FOCAL']['USE_CONV_EMBED'],    
            use_postln=cfg['BACKBONE']['FOCAL']['USE_POSTLN'],       
            use_postln_in_modulation=cfg['BACKBONE']['FOCAL']['USE_POSTLN_IN_MODULATION'], 
            scaling_modulator=scaling_modulator,
            use_layerscale=cfg['BACKBONE']['FOCAL']['USE_LAYERSCALE'], 
            use_checkpoint=use_checkpoint,
        )

        self._out_features = cfg['BACKBONE']['FOCAL']['OUT_FEATURES']

        self._out_feature_strides = {
            "res2": 4,
            "res3": 8,
            "res4": 16,
            "res5": 32,
        }
        self._out_feature_channels = {
            "res2": self.num_features[0],
            "res3": self.num_features[1],
            "res4": self.num_features[2],
            "res5": self.num_features[3],
        }

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.
        Returns:
            dict[str->Tensor]: names and the corresponding features
        """
        assert (
            x.dim() == 4
        ), f"SwinTransformer takes an input of shape (N, C, H, W). Got {x.shape} instead!"
        outputs = {}
        y = super().forward(x)
        for k in y.keys():
            if k in self._out_features:
                outputs[k] = y[k]
        return outputs

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

    @property
    def size_divisibility(self):
        return 32

@register_backbone
def get_focal_backbone(cfg):
    focal = D2FocalNet(cfg['MODEL'], 224)    

    if cfg['MODEL']['BACKBONE']['LOAD_PRETRAINED'] is True:
        filename = cfg['MODEL']['BACKBONE']['PRETRAINED']
        logger.info(f'=> init from {filename}')
        with PathManager.open(filename, "rb") as f:
            ckpt = torch.load(f)['model']
        focal.load_weights(ckpt, cfg['MODEL']['BACKBONE']['FOCAL'].get('PRETRAINED_LAYERS', ['*']), cfg['VERBOSE'])

    return focal