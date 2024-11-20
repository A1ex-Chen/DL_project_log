def _value_info2tensor_spec(value_info: onnx.ValueInfoProto):
    onnx_data_type_map = {'float': 'float32', 'double': 'float64'}
    elem_type_name = onnx.TensorProto.DataType.Name(value_info.type.
        tensor_type.elem_type).lower()
    dtype = onnx_data_type_map.get(elem_type_name, elem_type_name)

    def _get_dim(dim):
        which = dim.WhichOneof('value')
        if which is not None:
            dim = getattr(dim, which)
        return None if isinstance(dim, (str, bytes)) else dim
    shape = value_info.type.tensor_type.shape
    shape = tuple([_get_dim(d) for d in shape.dim])
    return TensorSpec(value_info.name, dtype=dtype, shape=shape)
