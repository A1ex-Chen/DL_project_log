# -*- coding: utf-8 -*-
"""
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>

Parts of this code are taken and adapted from:
https://github.com/hszhao/semseg/blob/master/model/pspnet.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
dist.init_process_group('gloo', init_method='file:///tmp/somefile', rank=0, world_size=1)
from src.models.model_utils import ConvBNAct
import numpy



class PyramidPoolingModule(nn.Module):



class AdaptivePyramidPoolingModule(nn.Module):


bn_mom = 0.1
BatchNorm2d = nn.SyncBatchNorm

class DAPPM(nn.Module):

    
class ConvBNReLU(nn.Sequential): #*ConvBN padding参数
        
# class PPContextModule(nn.Module):
#     """
#     Simple Context module.
#     Args:
#         in_channels (int): The number of input channels to pyramid pooling module.
#         inter_channels (int): The number of inter channels to pyramid pooling module.
#         out_channels (int): The number of output channels after pyramid pooling module.
#         bin_sizes (tuple, optional): The out size of pooled feature maps. Default: (1, 3).
#         align_corners (bool): An argument of F.interpolate. It should be set to False
#             when the output size of feature is even, e.g. 1024x512, otherwise it is True, e.g. 769x769.
#     """

#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  bins=(1,3), #bin_sizes=bins
#                  align_corners=False,
#                  upsampling_mode='bilinear'):
#         super().__init__()
#         # nn.ParameterList
#         # nn.ModuleList
#         inter_channels = in_channels // len(bins)      
#         self.upsampling_mode = upsampling_mode

#         #*nn.LayerList,nn指的是import paddle.nn as nn，nn.ModuleList
#         # *LayerList用于保存子层列表，它包含的子层将被正确地注册和添加。列表中的子层可以像常规python列表一样被索引。
#         self.stages = nn.ModuleList([
#             self._make_stage(in_channels, inter_channels, size)
#             for size in bins
#         ])
#         #The bin sizes of global-average-pooling are1 X 1,2 X 2 and 4 X 4 respectively.
        
#        #*在PaddlePaddle/PaddleSeg/paddleseg/models/layers/layer_libs.py里有，
#        #*需要导入from paddleseg.models import layers，增加layer_libs.py文件，该文件仍然存在import其它参数
#         self.conv_out = ConvBNReLU(
#          inter_channels,
#             out_channels,
#              bias=False,
#             kernel_size=3,
#             padding=1)

#         self.align_corners = align_corners

#     #*AdaptiveAvgPool2D ， PaddlePaddle/Paddle/python/paddle/nn/functional/pooling.py，新导入参数，很可能要导入新工具箱                                                                                  
#     def _make_stage(self, in_channels, out_channels, size):
#         prior = nn.AdaptiveAvgPool2d(output_size=size)  #nn.AdaptiveAvgPool2D
#         conv = ConvBNReLU(
#             in_channels, out_channels,bias=False, kernel_size=1)
#         return nn.Sequential(prior, conv)
#     #*需要import paddle，或者要明白paddle.shape是什么样的，直接写进来
#     def forward(self, input):
#         out = None
#         input_shape = numpy.shape(input)[2:]                                 

#         for stage in self.stages:
#             x = stage(input)
#             x = F.interpolate(
#                 x,
#                 input_shape,
#                 mode='bilinear',#*bilinear
#                 align_corners=self.align_corners)
            
        
#             if out is None:
#                 out = x
#             else:
#                 out += x

#         out = self.conv_out(out)
#         return out  
class PPContextModule(nn.Module):
        