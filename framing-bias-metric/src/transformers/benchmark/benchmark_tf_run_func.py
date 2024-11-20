def run_func(func):

    @wraps(func)
    def run_in_eager_mode(*args, **kwargs):
        return func(*args, **kwargs)

    @wraps(func)
    @tf.function(experimental_compile=use_xla)
    def run_in_graph_mode(*args, **kwargs):
        return func(*args, **kwargs)
    if do_eager_mode is True:
        assert use_xla is False, 'Cannot run model in XLA, if `args.eager_mode` is set to `True`. Please set `args.eager_mode=False`.'
        return run_in_eager_mode
    else:
        return run_in_graph_mode
