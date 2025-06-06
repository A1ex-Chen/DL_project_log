def configurable(init_func=None, *, from_config=None):
    """
    Decorate a function or a class's __init__ method so that it can be called
    with a :class:`CfgNode` object using a :func:`from_config` function that translates
    :class:`CfgNode` to arguments.

    Examples:
    ::
        # Usage 1: Decorator on __init__:
        class A:
            @configurable
            def __init__(self, a, b=2, c=3):
                pass

            @classmethod
            def from_config(cls, cfg):   # 'cfg' must be the first argument
                # Returns kwargs to be passed to __init__
                return {"a": cfg.A, "b": cfg.B}

        a1 = A(a=1, b=2)  # regular construction
        a2 = A(cfg)       # construct with a cfg
        a3 = A(cfg, b=3, c=4)  # construct with extra overwrite

        # Usage 2: Decorator on any function. Needs an extra from_config argument:
        @configurable(from_config=lambda cfg: {"a: cfg.A, "b": cfg.B})
        def a_func(a, b=2, c=3):
            pass

        a1 = a_func(a=1, b=2)  # regular call
        a2 = a_func(cfg)       # call with a cfg
        a3 = a_func(cfg, b=3, c=4)  # call with extra overwrite

    Args:
        init_func (callable): a class's ``__init__`` method in usage 1. The
            class must have a ``from_config`` classmethod which takes `cfg` as
            the first argument.
        from_config (callable): the from_config function in usage 2. It must take `cfg`
            as its first argument.
    """
    if init_func is not None:
        assert inspect.isfunction(init_func
            ) and from_config is None and init_func.__name__ == '__init__', 'Incorrect use of @configurable. Check API documentation for examples.'

        @functools.wraps(init_func)
        def wrapped(self, *args, **kwargs):
            try:
                from_config_func = type(self).from_config
            except AttributeError as e:
                raise AttributeError(
                    "Class with @configurable must have a 'from_config' classmethod."
                    ) from e
            if not inspect.ismethod(from_config_func):
                raise TypeError(
                    "Class with @configurable must have a 'from_config' classmethod."
                    )
            if _called_with_cfg(*args, **kwargs):
                explicit_args = _get_args_from_config(from_config_func, *
                    args, **kwargs)
                init_func(self, **explicit_args)
            else:
                init_func(self, *args, **kwargs)
        return wrapped
    else:
        if from_config is None:
            return configurable
        assert inspect.isfunction(from_config
            ), 'from_config argument of configurable must be a function!'

        def wrapper(orig_func):

            @functools.wraps(orig_func)
            def wrapped(*args, **kwargs):
                if _called_with_cfg(*args, **kwargs):
                    explicit_args = _get_args_from_config(from_config, *
                        args, **kwargs)
                    return orig_func(**explicit_args)
                else:
                    return orig_func(*args, **kwargs)
            wrapped.from_config = from_config
            return wrapped
        return wrapper
