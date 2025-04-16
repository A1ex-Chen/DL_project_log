def construct_init_net_from_params(params: Dict[str, Any], device_options:
    Optional[Dict[str, caffe2_pb2.DeviceOption]]=None) ->caffe2_pb2.NetDef:
    """
    Construct the init_net from params dictionary
    """
    init_net = caffe2_pb2.NetDef()
    device_options = device_options or {}
    for name, blob in params.items():
        if isinstance(blob, str):
            logger.warning(
                'Blob {} with type {} is not supported in generating init net, skipped.'
                .format(name, type(blob)))
            continue
        init_net.op.extend([create_const_fill_op(name, blob, device_option=
            device_options.get(name, None))])
        init_net.external_output.append(name)
    return init_net
