def _create_const_fill_op_from_numpy(name, tensor, device_option=None):
    assert type(tensor) == np.ndarray
    kTypeNameMapper = {np.dtype('float32'): 'GivenTensorFill', np.dtype(
        'int32'): 'GivenTensorIntFill', np.dtype('int64'):
        'GivenTensorInt64Fill', np.dtype('uint8'): 'GivenTensorStringFill'}
    args_dict = {}
    if tensor.dtype == np.dtype('uint8'):
        args_dict.update({'values': [str(tensor.data)], 'shape': [1]})
    else:
        args_dict.update({'values': tensor, 'shape': tensor.shape})
    if device_option is not None:
        args_dict['device_option'] = device_option
    return core.CreateOperator(kTypeNameMapper[tensor.dtype], [], [name],
        **args_dict)
