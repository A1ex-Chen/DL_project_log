def memory_usage(device):
    """Get GPU memory total and usage

    Parameters
    ----------
    device : torch.device

    Returns
    -------
    usage : namedtuple(torch.device, int, int)
        Total memory of the GPU and its current usage
    """
    if device.type == 'cpu':
        raise ValueError(
            f'Can only query GPU memory usage, but device is {device}')
    Usage = namedtuple('MemoryUsage', 'device total used')
    if device.type == 'cuda':
        device_idx = get_device_idx(device)
    try:
        total, used = os.popen(
            'nvidia-smi --query-gpu=memory.total,memory.used --format=csv,nounits,noheader'
            ).read().split('\n')[device_idx].split(',')
    except ValueError:
        print(
            f'Attempted to query CUDA device {device_idx}, does this system have that many GPUs?'
            )
    return Usage(device, int(total), int(used))
