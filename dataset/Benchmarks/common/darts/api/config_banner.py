def banner(device):
    """Print a banner of the system config

    Parameters
    ----------
    device : torch.device
    """
    print('=' * 80)
    info = get_torch_info()
    torch_msg = (f'Pytorch version: {info.torch_version} ',
        f'cuda version {info.cuda_version} ',
        f'cudnn version {info.cudnn_version}')
    print(''.join(torch_msg))
    if device.type == 'cuda':
        device_idx = get_device_idx(device)
        usage = memory_usage(device)
        print(f'CUDA Device name {torch.cuda.get_device_name(device_idx)}')
        print(f'CUDA memory - total: {usage.total} current usage: {usage.used}'
            )
    else:
        print(f'Using CPU')
    print(dtm.datetime.now().strftime('%Y/%m/%d - %H:%M:%S'))
    print('=' * 80)
