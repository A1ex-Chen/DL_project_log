import torch

from deepview_profile.profiler.backward import get_grad_fn, flatten_operation_output


class AutogradEngine:
    """
    Emulates the backward pass for a given model output, for timing purposes.
    """

    @classmethod
