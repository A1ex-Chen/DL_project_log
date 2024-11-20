import torch
from torch import nn
from yolov6.layers.common import RepBlock, RepVGGBlock, BottleRep, BepC3, ConvBNReLU, Transpose, BiFusion, \
                                MBLABlock, ConvBNHS, CSPBlock, DPBlock

# _QUANT=False
class RepPANNeck(nn.Module):
    """RepPANNeck Module
    EfficientRep is the default backbone of this model.
    RepPANNeck has the balance of feature fusion ability and hardware efficiency.
    """





class RepBiFPANNeck(nn.Module):
    """RepBiFPANNeck Module
    """
    # [64, 128, 256, 512, 1024]
    # [256, 128, 128, 256, 256, 512]





class RepPANNeck6(nn.Module):
    """RepPANNeck+P6 Module
    EfficientRep is the default backbone of this model.
    RepPANNeck has the balance of feature fusion ability and hardware efficiency.
    """
    # [64, 128, 256, 512, 768, 1024]
    # [512, 256, 128, 256, 512, 1024]




class RepBiFPANNeck6(nn.Module):
    """RepBiFPANNeck_P6 Module
    """
    # [64, 128, 256, 512, 768, 1024]
    # [512, 256, 128, 256, 512, 1024]





class CSPRepPANNeck(nn.Module):
    """
    CSPRepPANNeck module.
    """




class CSPRepBiFPANNeck(nn.Module):
    """
    CSPRepBiFPANNeck module.
    """





class CSPRepPANNeck_P6(nn.Module):
    """CSPRepPANNeck_P6 Module
    """
    # [64, 128, 256, 512, 768, 1024]
    # [512, 256, 128, 256, 512, 1024]




class CSPRepBiFPANNeck_P6(nn.Module):
    """CSPRepBiFPANNeck_P6 Module
    """
    # [64, 128, 256, 512, 768, 1024]
    # [512, 256, 128, 256, 512, 1024]



class Lite_EffiNeck(nn.Module):

