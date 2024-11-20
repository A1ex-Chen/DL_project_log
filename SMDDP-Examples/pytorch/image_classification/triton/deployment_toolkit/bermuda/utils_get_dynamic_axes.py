def get_dynamic_axes(dataloader, batch_size_dim=0):
    input_shapes, output_shapes = get_shapes_with_dynamic_axes(dataloader,
        batch_size_dim)
    all_shapes = {**input_shapes, **output_shapes}
    dynamic_axes = {}
    for k, shape in all_shapes.items():
        for idx, s in enumerate(shape):
            if s == -1:
                dynamic_axes[k] = {idx: k + '_' + str(idx)}
    for k, v in all_shapes.items():
        if k in dynamic_axes:
            dynamic_axes[k].update({batch_size_dim: 'batch_size_' + str(
                batch_size_dim)})
        else:
            dynamic_axes[k] = {batch_size_dim: 'batch_size_' + str(
                batch_size_dim)}
    return dynamic_axes
