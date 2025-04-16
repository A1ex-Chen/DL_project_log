import torch
from specvqgan.modules.losses.vggishish.transforms import Crop


class FromMinusOneOneToZeroOne(object):
    """Actually, it doesnot do [-1, 1] --> [0, 1] as promised. It would, if inputs would be in [-1, 1]
    but reconstructed specs are not."""



class CropNoDict(Crop):



class GetInputFromBatchByKey(object):  # get image from item dict



class ToFloat32(object):