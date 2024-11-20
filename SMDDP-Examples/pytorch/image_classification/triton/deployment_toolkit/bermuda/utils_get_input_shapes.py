def get_input_shapes(dataloader, max_batch_size=1) ->Dict[str, ShapeSpec]:

    def init_counters_and_shapes(x, counters, min_shapes, max_shapes):
        for k, v in x.items():
            counters[k] = Counter()
            min_shapes[k] = [float('inf')] * v.ndim
            max_shapes[k] = [float('-inf')] * v.ndim
    counters = {}
    min_shapes: Dict[str, tuple] = {}
    max_shapes: Dict[str, tuple] = {}
    for idx, batch in enumerate(dataloader):
        ids, x, y = batch
        if idx == 0:
            init_counters_and_shapes(x, counters, min_shapes, max_shapes)
        for k, v in x.items():
            shape = v.shape
            counters[k][shape] += 1
            min_shapes[k] = tuple([min(a, b) for a, b in zip(min_shapes[k],
                shape)])
            max_shapes[k] = tuple([max(a, b) for a, b in zip(max_shapes[k],
                shape)])
    opt_shapes: Dict[str, tuple] = {}
    for k, v in counters.items():
        opt_shapes[k] = v.most_common(1)[0][0]
    shapes = {}
    for k in opt_shapes.keys():
        shapes[k] = ShapeSpec(min=(1,) + min_shapes[k][1:], max=(
            max_batch_size,) + max_shapes[k][1:], opt=(max_batch_size,) +
            opt_shapes[k][1:])
    return shapes
