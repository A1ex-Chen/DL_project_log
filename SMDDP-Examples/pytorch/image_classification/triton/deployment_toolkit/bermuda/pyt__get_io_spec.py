def _get_io_spec(model, dataloader_fn):
    precision = model.precision
    dataloader = dataloader_fn()
    input_dtypes, output_dtypes = _get_tensor_dtypes(dataloader, precision)
    input_shapes, output_shapes = get_shapes_with_dynamic_axes(dataloader)
    inputs = {name: TensorSpec(name=name, dtype=input_dtypes[name], shape=
        tuple(input_shapes[name])) for name in model.inputs}
    outputs = {name: TensorSpec(name=name, dtype=output_dtypes[name], shape
        =tuple(output_shapes[name])) for name in model.outputs}
    return InputOutputSpec(inputs, outputs)
