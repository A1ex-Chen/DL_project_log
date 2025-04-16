def maybe_to_cpu(x):
    try:
        like_gpu_tensor = x.device.type == 'cuda' and hasattr(x, 'to')
    except AttributeError:
        like_gpu_tensor = False
    if like_gpu_tensor:
        return x.to(device='cpu')
    else:
        return x
