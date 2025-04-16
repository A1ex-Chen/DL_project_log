def benchmark_fn(f, *args, **kwargs):
    t0 = benchmark.Timer(stmt='f(*args, **kwargs)', globals={'args': args,
        'kwargs': kwargs, 'f': f}, num_threads=torch.get_num_threads())
    return f'{t0.blocked_autorange().mean:.3f}'
