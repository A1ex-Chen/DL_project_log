def init_counters_and_shapes(x, counters, min_shapes, max_shapes):
    for k, v in x.items():
        counters[k] = Counter()
        min_shapes[k] = [float('inf')] * v.ndim
        max_shapes[k] = [float('-inf')] * v.ndim
