@wraps(func)
@tf.function(experimental_compile=use_xla)
def run_in_graph_mode(*args, **kwargs):
    return func(*args, **kwargs)
