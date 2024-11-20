@staticmethod
def from_name(name: Name) ->Type['Algorithm']:
    if name == Algorithm.Name.MASK_RCNN:
        from .mask_rcnn import MaskRCNN as T
    else:
        raise ValueError('Invalid algorithm name')
    return T
