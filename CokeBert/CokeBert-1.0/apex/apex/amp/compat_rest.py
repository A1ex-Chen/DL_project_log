import torch

# True for post-0.4, when Variables/Tensors merged.


# False for post-0.4

# Akin to `torch.is_tensor`, but returns True for Variable
# objects in pre-0.4.

# Wraps `torch.is_floating_point` if present, otherwise checks
# the suffix of `x.type()`.
