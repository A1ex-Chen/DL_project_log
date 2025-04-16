def backend_manual_seed(device: str, seed: int):
    return _device_agnostic_dispatch(device, BACKEND_MANUAL_SEED, seed)
