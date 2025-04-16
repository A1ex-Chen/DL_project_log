def torch_save(*args, use_dill=True, **kwargs):
    """
    Optionally use dill to serialize lambda functions where pickle does not, adding robustness with 3 retries and
    exponential standoff in case of save failure.

    Args:
        *args (tuple): Positional arguments to pass to torch.save.
        use_dill (bool): Whether to try using dill for serialization if available. Defaults to True.
        **kwargs (any): Keyword arguments to pass to torch.save.
    """
    try:
        assert use_dill
        import dill as pickle
    except (AssertionError, ImportError):
        import pickle
    if 'pickle_module' not in kwargs:
        kwargs['pickle_module'] = pickle
    for i in range(4):
        try:
            return _torch_save(*args, **kwargs)
        except RuntimeError as e:
            if i == 3:
                raise e
            time.sleep(2 ** i / 2)
