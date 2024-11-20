def torch_only_method(fn):

    def wrapper(*args, **kwargs):
        if not _torch_available:
            raise ImportError(
                'You need to install pytorch to use this method or class, or activate it with environment variables USE_TORCH=1 and USE_TF=0.'
                )
        else:
            return fn(*args, **kwargs)
    return wrapper
