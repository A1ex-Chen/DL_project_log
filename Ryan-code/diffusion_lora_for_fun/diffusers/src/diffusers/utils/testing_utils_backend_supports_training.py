def backend_supports_training(device: str):
    if not is_torch_available():
        return False
    if device not in BACKEND_SUPPORTS_TRAINING:
        device = 'default'
    return BACKEND_SUPPORTS_TRAINING[device]
