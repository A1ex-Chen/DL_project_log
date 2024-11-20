def pool_tensor(self, tensor, mode='mean', stride=2):
    """Apply 1D pooling to a tensor of size [B x T (x H)]."""
    if tensor is None:
        return None
    if isinstance(tensor, (tuple, list)):
        return type(tensor)(self.pool_tensor(tensor, mode=mode, stride=
            stride) for x in tensor)
    if self.config.separate_cls:
        suffix = tensor[:, :-1] if self.config.truncate_seq else tensor
        tensor = torch.cat([tensor[:, :1], suffix], dim=1)
    ndim = tensor.ndim
    if ndim == 2:
        tensor = tensor[:, None, :, None]
    elif ndim == 3:
        tensor = tensor[:, None, :, :]
    stride = stride, 1
    if mode == 'mean':
        tensor = F.avg_pool2d(tensor, stride, stride=stride, ceil_mode=True)
    elif mode == 'max':
        tensor = F.max_pool2d(tensor, stride, stride=stride, ceil_mode=True)
    elif mode == 'min':
        tensor = -F.max_pool2d(-tensor, stride, stride=stride, ceil_mode=True)
    else:
        raise NotImplementedError(
            "The supported modes are 'mean', 'max' and 'min'.")
    if ndim == 2:
        return tensor[:, 0, :, 0]
    elif ndim == 3:
        return tensor[:, 0]
    return tensor
