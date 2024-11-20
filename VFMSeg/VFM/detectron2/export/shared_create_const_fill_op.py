def create_const_fill_op(name: str, blob: Union[np.ndarray, workspace.
    Int8Tensor], device_option: Optional[caffe2_pb2.DeviceOption]=None
    ) ->caffe2_pb2.OperatorDef:
    """
    Given a blob object, return the Caffe2 operator that creates this blob
    as constant. Currently support NumPy tensor and Caffe2 Int8Tensor.
    """
    tensor_type = type(blob)
    assert tensor_type in [np.ndarray, workspace.Int8Tensor
        ], 'Error when creating const fill op for "{}", unsupported blob type: {}'.format(
        name, type(blob))
    if tensor_type == np.ndarray:
        return _create_const_fill_op_from_numpy(name, blob, device_option)
    elif tensor_type == workspace.Int8Tensor:
        assert device_option is None
        return _create_const_fill_op_from_c2_int8_tensor(name, blob)
