def get_mask(tensor):
    """ Returns mask of non-illegal values for tensor.

    Args:
        tensor (tensor): Numpy or Pytorch tensor
    """
    tensor, is_numpy = to_pytorch(tensor, True)
    mask = (abs(tensor) != np.inf) & (torch.isnan(tensor) == False)
    mask = mask.bool()
    if is_numpy:
        mask = mask.numpy()
    return mask
