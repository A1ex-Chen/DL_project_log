from abc import ABCMeta
from abc import abstractmethod
from abc import abstractproperty
from tDBN.core import box_np_ops
from tDBN.core import box_torch_ops
import torch

class BoxCoder(object):
    """Abstract base class for box coder."""
    __metaclass__ = ABCMeta

    @abstractproperty



    @abstractmethod

    @abstractmethod

class GroundBox3dCoder(BoxCoder):

    @property



class BevBoxCoder(BoxCoder):
    """WARNING: this coder will return encoding with size=5, but
    takes size=7 boxes, anchors
    """

    @property



class GroundBox3dCoderTorch(GroundBox3dCoder):


class BevBoxCoderTorch(BevBoxCoder):


