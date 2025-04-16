import torch
from torch.nn.parameter import Parameter
import sys
class Reparameterization(object):
    """
    Class interface for performing weight reparameterizations
    Arguments:
        name (str): name of weight parameter
        dim (int): dimension over which to compute the norm
        module (nn.Module): parent module to which param `name` is registered to
        retain_forward (bool, optional): if False deletes weight on call to 
            module.backward. Used to avoid memory leaks with DataParallel Default: True
    Attributes:
        reparameterization_names (list, str): contains names of all parameters 
            needed to compute reparameterization.
        backward_hook_key (int): torch.utils.hooks.RemovableHandle.id for hook used in module backward pass.
    """




    @staticmethod

    @staticmethod



