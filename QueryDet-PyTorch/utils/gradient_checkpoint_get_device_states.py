def get_device_states(*args) ->Tuple[List[int], List[torch.Tensor]]:
    fwd_gpu_devices = list(set(arg.get_device() for arg in args if 
        isinstance(arg, torch.Tensor) and arg.is_cuda))
    fwd_gpu_states = []
    for device in fwd_gpu_devices:
        with torch.cuda.device(device):
            fwd_gpu_states.append(torch.cuda.get_rng_state())
    return fwd_gpu_devices, fwd_gpu_states
