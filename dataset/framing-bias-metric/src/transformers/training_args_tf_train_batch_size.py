@property
def train_batch_size(self) ->int:
    """
        The actual batch size for training (may differ from :obj:`per_gpu_train_batch_size` in distributed training).
        """
    if self.per_gpu_train_batch_size:
        logger.warning(
            'Using deprecated `--per_gpu_train_batch_size` argument which will be removed in a future version. Using `--per_device_train_batch_size` is preferred.'
            )
    per_device_batch_size = (self.per_gpu_train_batch_size or self.
        per_device_train_batch_size)
    return per_device_batch_size * self.n_replicas
