@staticmethod
def from_name(name: Name) ->Type['Backbone']:
    if name == Backbone.Name.MOBILENET_V3_SMALL:
        from .mobilenet_v3_small import MobileNet_v3_Small as T
    elif name == Backbone.Name.MOBILENET_V3_LARGE:
        from .mobilenet_v3_large import MobileNet_v3_Large as T
    elif name == Backbone.Name.RESNET18:
        from .resnet18 import ResNet18 as T
    elif name == Backbone.Name.RESNET34:
        from .resnet34 import ResNet34 as T
    elif name == Backbone.Name.RESNET50:
        from .resnet50 import ResNet50 as T
    elif name == Backbone.Name.RESNET101:
        from .resnet101 import ResNet101 as T
    elif name == Backbone.Name.RESNET152:
        from .resnet152 import ResNet152 as T
    elif name == Backbone.Name.RESNEXT50_32X4D:
        from .resnext50_32x4d import ResNeXt50_32x4d as T
    elif name == Backbone.Name.RESNEXT101_32X8D:
        from .resnext101_32x8d import ResNeXt101_32x8d as T
    elif name == Backbone.Name.WIDE_RESNET50_2:
        from .wide_resnet50_2 import WideResNet50_2 as T
    elif name == Backbone.Name.WIDE_RESNET101_2:
        from .wide_resnet101_2 import WideResNet101_2 as T
    elif name == Backbone.Name.SENET154:
        from .senet154 import SENet154 as T
    elif name == Backbone.Name.SE_RESNEXT50_32X4D:
        from .se_resnext50_32x4d import SEResNeXt50_32x4d as T
    elif name == Backbone.Name.SE_RESNEXT101_32X4D:
        from .se_resnext101_32x4d import SEResNeXt101_32x4d as T
    elif name == Backbone.Name.NASNET_A_LARGE:
        from .nasnet_a_large import NASNet_A_Large as T
    elif name == Backbone.Name.PNASNET_5_LARGE:
        from .pnasnet_5_large import PNASNet_5_Large as T
    elif name == Backbone.Name.RESNEST50:
        from .resnest50 import ResNeSt50 as T
    elif name == Backbone.Name.RESNEST101:
        from .resnest101 import ResNeSt101 as T
    elif name == Backbone.Name.RESNEST200:
        from .resnest200 import ResNeSt200 as T
    elif name == Backbone.Name.RESNEST269:
        from .resnest269 import ResNeSt269 as T
    elif name == Backbone.Name.RegNet_Y_8GF:
        from .regnet_y_8gf import RegNet_Y_8GF as T
    elif name == Backbone.Name.EFFICIENTNET_B7:
        from .efficientnet_b7 import EfficientNet_B7 as T
    elif name == Backbone.Name.ConvNeXt_Base:
        from .convnext_base import ConvNeXt_Base as T
    elif name == Backbone.Name.EfficientNet_V2:
        from .efficientnet_v2 import EfficientNet_V2 as T
    else:
        raise ValueError('Invalid backbone name')
    return T
