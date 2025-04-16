from pickle import FALSE
from torch import nn
from yolov6.layers.common import BottleRep, RepVGGBlock, RepBlock, BepC3, SimSPPF, SPPF, SimCSPSPPF, CSPSPPF, ConvBNSiLU, \
                                MBLABlock, ConvBNHS, Lite_EffiBlockS2, Lite_EffiBlockS1


class EfficientRep(nn.Module):
    '''EfficientRep Backbone
    EfficientRep is handcrafted by hardware-aware neural network design.
    With rep-style struct, EfficientRep is friendly to high-computation hardware(e.g. GPU).
    '''




class EfficientRep6(nn.Module):
    '''EfficientRep+P6 Backbone
    EfficientRep is handcrafted by hardware-aware neural network design.
    With rep-style struct, EfficientRep is friendly to high-computation hardware(e.g. GPU).
    '''




class CSPBepBackbone(nn.Module):
    """
    CSPBepBackbone module.
    """




class CSPBepBackbone_P6(nn.Module):
    """
    CSPBepBackbone+P6 module.
    """



class Lite_EffiBackbone(nn.Module):


    @staticmethod