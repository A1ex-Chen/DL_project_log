def get_shapes_with_dynamic_axes(dataloader, batch_size_dim=0):

    def _set_dynamic_shapes(t, shapes):
        for k, v in t.items():
            shape = list(v.shape)
            for dim, s in enumerate(shape):
                if shapes[k][dim] != -1 and shapes[k][dim] != s:
                    shapes[k][dim] = -1
    input_shapes = {}
    output_shapes = {}
    for batch in dataloader:
        _, x, y = batch
        for k, v in x.items():
            input_shapes[k] = list(v.shape)
        for k, v in y.items():
            output_shapes[k] = list(v.shape)
        break
    max_num_iters = 100
    for idx, batch in enumerate(dataloader):
        if idx >= max_num_iters:
            break
        _, x, y = batch
        _set_dynamic_shapes(x, input_shapes)
        _set_dynamic_shapes(y, output_shapes)
    return input_shapes, output_shapes
