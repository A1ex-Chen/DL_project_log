# https://github.com/meituan/YOLOv6/

from torch import nn

from deeplite_torch_zoo.src.object_detection.yolo.flexible_yolo.yolov6.layers.common import (
    CSPSPPF,
    SPPF,
    BepC3,
    ConvBNHS,
    ConvBNSiLU,
    Lite_EffiBlockS1,
    Lite_EffiBlockS2,
    MBLABlock,
    RepBlock,
    RepVGGBlock,
    SimCSPSPPF,
    SimSPPF,
)


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