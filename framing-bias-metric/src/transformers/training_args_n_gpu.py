@property
@torch_required
def n_gpu(self):
    """
        The number of GPUs used by this process.

        Note:
            This will only be greater than one when you have multiple GPUs available but are not using distributed
            training. For distributed training, it will always be 1.
        """
    return self._setup_devices[1]
