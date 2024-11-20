def select_device(device='', batch_size=0, newline=True):
    s = (
        f'YOLOv5 🚀 {git_describe() or file_date()} Python-{platform.python_version()} torch-{torch.__version__} '
        )
    device = str(device).strip().lower().replace('cuda:', '').replace('none',
        '')
    cpu = device == 'cpu'
    mps = device == 'mps'
    if cpu or mps:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    elif device:
        os.environ['CUDA_VISIBLE_DEVICES'] = device
        assert torch.cuda.is_available() and torch.cuda.device_count() >= len(
            device.replace(',', '')
            ), f"Invalid CUDA '--device {device}' requested, use '--device cpu' or pass valid CUDA device(s)"
    if not (cpu or mps) and torch.cuda.is_available():
        devices = device.split(',') if device else '0'
        n = len(devices)
        if n > 1 and batch_size > 0:
            assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
        space = ' ' * (len(s) + 1)
        for i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
            s += f"""{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / (1 << 20):.0f}MiB)
"""
        arg = 'cuda:0'
    elif mps and getattr(torch, 'has_mps', False
        ) and torch.backends.mps.is_available():
        s += 'MPS\n'
        arg = 'mps'
    else:
        s += 'CPU\n'
        arg = 'cpu'
    if not newline:
        s = s.rstrip()
    LOGGER.info(s)
    return torch.device(arg)
