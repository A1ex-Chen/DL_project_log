def _create_const_fill_op_from_c2_int8_tensor(name, int8_tensor):
    assert type(int8_tensor) == workspace.Int8Tensor
    kTypeNameMapper = {np.dtype('int32'): 'Int8GivenIntTensorFill', np.
        dtype('uint8'): 'Int8GivenTensorFill'}
    tensor = int8_tensor.data
    assert tensor.dtype in [np.dtype('uint8'), np.dtype('int32')]
    values = tensor.tobytes() if tensor.dtype == np.dtype('uint8') else tensor
    return core.CreateOperator(kTypeNameMapper[tensor.dtype], [], [name],
        values=values, shape=tensor.shape, Y_scale=int8_tensor.scale,
        Y_zero_point=int8_tensor.zero_point)
