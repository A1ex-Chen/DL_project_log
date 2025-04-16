def _get_dtype(vi):
    t = vi.type
    if hasattr(t, 'tensor_type'):
        type_id = t.tensor_type.elem_type
    else:
        raise NotImplementedError('Not implemented yet')
    return TENSOR_TYPE_TO_NP_TYPE[type_id]
