def get_device_idx(device):
    """Get the CUDA device from torch

    Parameters
    ----------
    device : torch.device

    Returns
    -------
    index of the CUDA device
    """
    return 0 if device.index is None else device.index
