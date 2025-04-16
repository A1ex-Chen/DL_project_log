def _clone_tensors(self, argument, for_inplace):
    if isinstance(argument, torch.Tensor):
        detached = argument.detach()
        detached.requires_grad_(argument.requires_grad)
        return detached if not for_inplace else detached.clone()
    if isinstance(argument, tuple):
        return tuple(map(lambda arg: self._clone_tensors(arg, for_inplace),
            argument))
    if isinstance(argument, list):
        return list(map(lambda arg: self._clone_tensors(arg, for_inplace),
            argument))
    return argument
