@staticmethod
def from_name(name: Name) ->Type['Algorithm']:
    if name == Algorithm.Name.MOBILENET_V2:
        from .mobilenet_v2 import MobileNet_v2 as T
    elif name == Algorithm.Name.GOOGLENET:
        from .googlenet import GoogLeNet as T
    elif name == Algorithm.Name.INCEPTION_V3:
        from .inception_v3 import Inception_v3 as T
    elif name == Algorithm.Name.RESNET18:
        from .resnet18 import ResNet18 as T
    elif name == Algorithm.Name.RESNET34:
        from .resnet34 import ResNet34 as T
    elif name == Algorithm.Name.RESNET50:
        from .resnet50 import ResNet50 as T
    elif name == Algorithm.Name.RESNET101:
        from .resnet101 import ResNet101 as T
    elif name == Algorithm.Name.EFFICIENTNET_B0:
        from .efficientnet_b0 import EfficientNet_B0 as T
    elif name == Algorithm.Name.EFFICIENTNET_B1:
        from .efficientnet_b1 import EfficientNet_B1 as T
    elif name == Algorithm.Name.EFFICIENTNET_B2:
        from .efficientnet_b2 import EfficientNet_B2 as T
    elif name == Algorithm.Name.EFFICIENTNET_B3:
        from .efficientnet_b3 import EfficientNet_B3 as T
    elif name == Algorithm.Name.EFFICIENTNET_B4:
        from .efficientnet_b4 import EfficientNet_B4 as T
    elif name == Algorithm.Name.EFFICIENTNET_B5:
        from .efficientnet_b5 import EfficientNet_B5 as T
    elif name == Algorithm.Name.EFFICIENTNET_B6:
        from .efficientnet_b6 import EfficientNet_B6 as T
    elif name == Algorithm.Name.EFFICIENTNET_B7:
        from .efficientnet_b7 import EfficientNet_B7 as T
    elif name == Algorithm.Name.RESNEST50:
        from .resnest50 import ResNeSt50 as T
    elif name == Algorithm.Name.RESNEST101:
        from .resnest101 import ResNeSt101 as T
    elif name == Algorithm.Name.RESNEST200:
        from .resnest200 import ResNeSt200 as T
    elif name == Algorithm.Name.RESNEST269:
        from .resnest269 import ResNeSt269 as T
    elif name == Algorithm.Name.REGNET_Y_400MF:
        from .regnet_y_400mf import RegNet_Y_400MF as T
    else:
        raise ValueError('Invalid algorithm name')
    return T
