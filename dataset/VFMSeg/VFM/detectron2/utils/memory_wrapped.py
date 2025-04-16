@wraps(func)
def wrapped(*args, **kwargs):
    with _ignore_torch_cuda_oom():
        return func(*args, **kwargs)
    torch.cuda.empty_cache()
    with _ignore_torch_cuda_oom():
        return func(*args, **kwargs)
    logger = logging.getLogger(__name__)
    logger.info('Attempting to copy inputs of {} to CPU due to CUDA OOM'.
        format(str(func)))
    new_args = (maybe_to_cpu(x) for x in args)
    new_kwargs = {k: maybe_to_cpu(v) for k, v in kwargs.items()}
    return func(*new_args, **new_kwargs)
