@staticmethod
def _single_check(im, stride=32):
    """Validate and format an image to torch.Tensor."""
    s = (
        f'WARNING ⚠️ torch.Tensor inputs should be BCHW i.e. shape(1, 3, 640, 640) divisible by stride {stride}. Input shape{tuple(im.shape)} is incompatible.'
        )
    if len(im.shape) != 4:
        if len(im.shape) != 3:
            raise ValueError(s)
        LOGGER.warning(s)
        im = im.unsqueeze(0)
    if im.shape[2] % stride or im.shape[3] % stride:
        raise ValueError(s)
    if im.max() > 1.0 + torch.finfo(im.dtype).eps:
        LOGGER.warning(
            f'WARNING ⚠️ torch.Tensor inputs should be normalized 0.0-1.0 but max value is {im.max()}. Dividing input by 255.'
            )
        im = im.float() / 255.0
    return im
