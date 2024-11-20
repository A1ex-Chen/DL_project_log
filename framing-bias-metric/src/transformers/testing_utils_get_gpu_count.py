def get_gpu_count():
    """
    Return the number of available gpus (regardless of whether torch or tf is used)
    """
    if _torch_available:
        import torch
        return torch.cuda.device_count()
    elif _tf_available:
        import tensorflow as tf
        return len(tf.config.list_physical_devices('GPU'))
    else:
        return 0
