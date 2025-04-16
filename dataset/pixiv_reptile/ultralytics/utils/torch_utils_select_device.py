def select_device(device='', batch=0, newline=False, verbose=True):
    """
    Selects the appropriate PyTorch device based on the provided arguments.

    The function takes a string specifying the device or a torch.device object and returns a torch.device object
    representing the selected device. The function also validates the number of available devices and raises an
    exception if the requested device(s) are not available.

    Args:
        device (str | torch.device, optional): Device string or torch.device object.
            Options are 'None', 'cpu', or 'cuda', or '0' or '0,1,2,3'. Defaults to an empty string, which auto-selects
            the first available GPU, or CPU if no GPU is available.
        batch (int, optional): Batch size being used in your model. Defaults to 0.
        newline (bool, optional): If True, adds a newline at the end of the log string. Defaults to False.
        verbose (bool, optional): If True, logs the device information. Defaults to True.

    Returns:
        (torch.device): Selected device.

    Raises:
        ValueError: If the specified device is not available or if the batch size is not a multiple of the number of
            devices when using multiple GPUs.

    Examples:
        >>> select_device('cuda:0')
        device(type='cuda', index=0)

        >>> select_device('cpu')
        device(type='cpu')

    Note:
        Sets the 'CUDA_VISIBLE_DEVICES' environment variable for specifying which GPUs to use.
    """
    if isinstance(device, torch.device):
        return device
    s = (
        f'Ultralytics YOLOv{__version__} 🚀 Python-{PYTHON_VERSION} torch-{torch.__version__} '
        )
    device = str(device).lower()
    for remove in ('cuda:', 'none', '(', ')', '[', ']', "'", ' '):
        device = device.replace(remove, '')
    cpu = device == 'cpu'
    mps = device in {'mps', 'mps:0'}
    if cpu or mps:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    elif device:
        if device == 'cuda':
            device = '0'
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)
        os.environ['CUDA_VISIBLE_DEVICES'] = device
        if not (torch.cuda.is_available() and torch.cuda.device_count() >=
            len(device.split(','))):
            LOGGER.info(s)
            install = (
                """See https://pytorch.org/get-started/locally/ for up-to-date torch install instructions if no CUDA devices are seen by torch.
"""
                 if torch.cuda.device_count() == 0 else '')
            raise ValueError(
                f"""Invalid CUDA 'device={device}' requested. Use 'device=cpu' or pass valid CUDA device(s) if available, i.e. 'device=0' or 'device=0,1,2,3' for Multi-GPU.

torch.cuda.is_available(): {torch.cuda.is_available()}
torch.cuda.device_count(): {torch.cuda.device_count()}
os.environ['CUDA_VISIBLE_DEVICES']: {visible}
{install}"""
                )
    if not cpu and not mps and torch.cuda.is_available():
        devices = device.split(',') if device else '0'
        n = len(devices)
        if n > 1:
            if batch < 1:
                raise ValueError(
                    'AutoBatch with batch<1 not supported for Multi-GPU training, please specify a valid batch size, i.e. batch=16.'
                    )
            if batch >= 0 and batch % n != 0:
                raise ValueError(
                    f"'batch={batch}' must be a multiple of GPU count {n}. Try 'batch={batch // n * n}' or 'batch={batch // n * n + n}', the nearest batch sizes evenly divisible by {n}."
                    )
        space = ' ' * (len(s) + 1)
        for i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
            s += f"""{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / (1 << 20):.0f}MiB)
"""
        arg = 'cuda:0'
    elif mps and TORCH_2_0 and torch.backends.mps.is_available():
        s += f'MPS ({get_cpu_info()})\n'
        arg = 'mps'
    else:
        s += f'CPU ({get_cpu_info()})\n'
        arg = 'cpu'
    if verbose:
        LOGGER.info(s if newline else s.rstrip())
    return torch.device(arg)
