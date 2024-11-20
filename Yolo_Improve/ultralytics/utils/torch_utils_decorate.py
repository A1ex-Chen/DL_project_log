def decorate(fn):
    """Applies appropriate torch decorator for inference mode based on torch version."""
    if TORCH_1_9 and torch.is_inference_mode_enabled():
        return fn
    else:
        return (torch.inference_mode if TORCH_1_9 else torch.no_grad)()(fn)
