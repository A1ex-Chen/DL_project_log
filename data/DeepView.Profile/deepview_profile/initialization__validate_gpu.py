def _validate_gpu():
    import torch
    if not torch.cuda.is_available():
        logger.error(
            'DeepView did not detect a GPU on this machine. DeepView only profiles deep learning workloads on GPUs.'
            )
        return False
    return True
