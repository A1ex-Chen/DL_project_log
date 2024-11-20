def select_device(device='', batch=0, newline=False, verbose=True):
    """Selects PyTorch Device. Options are device = None or 'cpu' or 0 or '0' or '0,1,2,3'."""
    s = ''
    device = str(device).lower()
    for remove in ('cuda:', 'none', '(', ')', '[', ']', "'", ' '):
        device = device.replace(remove, '')
    cpu = device == 'cpu'
    mps = device == 'mps'
    if cpu or mps:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    elif device:
        if device == 'cuda':
            device = '0'
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)
        os.environ['CUDA_VISIBLE_DEVICES'] = device
        if not (torch.cuda.is_available() and torch.cuda.device_count() >=
            len(device.replace(',', ''))):
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
        if n > 1 and batch > 0 and batch % n != 0:
            raise ValueError(
                f"'batch={batch}' must be a multiple of GPU count {n}. Try 'batch={batch // n * n}' or 'batch={batch // n * n + n}', the nearest batch sizes evenly divisible by {n}."
                )
        space = ' ' * (len(s) + 1)
        for i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
            s += f"""{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / (1 << 20):.0f}MiB)
"""
        arg = 'cuda:0'
    elif mps and getattr(torch, 'has_mps', False
        ) and torch.backends.mps.is_available() and TORCH_2_0:
        s += 'MPS\n'
        arg = 'mps'
    else:
        s += 'CPU\n'
        arg = 'cpu'
    if verbose and RANK == -1:
        LOGGER.info(s if newline else s.rstrip())
    return torch.device(arg)
