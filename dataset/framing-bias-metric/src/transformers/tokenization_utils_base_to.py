@torch_required
def to(self, device: Union[str, 'torch.device']) ->'BatchEncoding':
    """
        Send all values to device by calling :obj:`v.to(device)` (PyTorch only).

        Args:
            device (:obj:`str` or :obj:`torch.device`): The device to put the tensors on.

        Returns:
            :class:`~transformers.BatchEncoding`: The same instance of :class:`~transformers.BatchEncoding` after
            modification.
        """
    if isinstance(device, str) or isinstance(device, torch.device):
        self.data = {k: v.to(device=device) for k, v in self.data.items()}
    else:
        logger.warning(
            f'Attempting to cast a BatchEncoding to another type, {str(device)}. This is not supported.'
            )
    return self
