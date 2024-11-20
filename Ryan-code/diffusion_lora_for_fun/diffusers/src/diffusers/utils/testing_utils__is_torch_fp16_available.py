def _is_torch_fp16_available(device):
    if not is_torch_available():
        return False
    import torch
    device = torch.device(device)
    try:
        x = torch.zeros((2, 2), dtype=torch.float16).to(device)
        _ = torch.mul(x, x)
        return True
    except Exception as e:
        if device.type == 'cuda':
            raise ValueError(
                f"You have passed a device of type 'cuda' which should work with 'fp16', but 'cuda' does not seem to be correctly installed on your machine: {e}"
                )
        return False
