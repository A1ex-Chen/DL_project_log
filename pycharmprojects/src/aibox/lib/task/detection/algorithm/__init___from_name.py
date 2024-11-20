@staticmethod
def from_name(name: Name) ->Type['Algorithm']:
    if name == Algorithm.Name.FASTER_RCNN:
        from .faster_rcnn import FasterRCNN as T
    elif name == Algorithm.Name.FPN:
        from .fpn import FPN as T
    elif name == Algorithm.Name.TORCH_FPN:
        from .torch_fpn import TorchFPN as T
    else:
        raise ValueError('Invalid algorithm name')
    return T
