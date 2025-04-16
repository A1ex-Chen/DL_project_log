def select_device(device='', batch_size=None):
    s = (
        f'YOLOR 🚀 {git_describe() or date_modified()} torch {torch.__version__} '
        )
    cpu = device.lower() == 'cpu'
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    elif device:
        os.environ['CUDA_VISIBLE_DEVICES'] = device
        assert torch.cuda.is_available(
            ), f'CUDA unavailable, invalid device {device} requested'
    cuda = not cpu and torch.cuda.is_available()
    if cuda:
        n = torch.cuda.device_count()
        if n > 1 and batch_size:
            assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
        space = ' ' * len(s)
        for i, d in enumerate(device.split(',') if device else range(n)):
            p = torch.cuda.get_device_properties(i)
            s += (
                f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / 1024 ** 2}MB)\n"
                )
    else:
        s += 'CPU\n'
    logger.info(s.encode().decode('ascii', 'ignore') if platform.system() ==
        'Windows' else s)
    return torch.device('cuda:0' if cuda else 'cpu')
