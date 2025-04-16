@property
def eval_batch_size(self) ->int:
    """
        The actual batch size for evaluation (may differ from :obj:`per_gpu_eval_batch_size` in distributed training).
        """
    if self.per_gpu_eval_batch_size:
        logger.warning(
            'Using deprecated `--per_gpu_eval_batch_size` argument which will be removed in a future version. Using `--per_device_eval_batch_size` is preferred.'
            )
    per_device_batch_size = (self.per_gpu_eval_batch_size or self.
        per_device_eval_batch_size)
    if not self.model_parallel:
        eval_batch_size = per_device_batch_size * max(1, self.n_gpu)
    else:
        eval_batch_size = per_device_batch_size
    return eval_batch_size
