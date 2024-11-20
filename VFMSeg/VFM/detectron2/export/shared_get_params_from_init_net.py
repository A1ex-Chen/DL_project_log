def get_params_from_init_net(init_net: caffe2_pb2.NetDef) ->[Dict[str, Any],
    Dict[str, caffe2_pb2.DeviceOption]]:
    """
    Take the output blobs from init_net by running it.
    Outputs:
        params: dict from blob name to numpy array
        device_options: dict from blob name to the device option of its creating op
    """

    def _get_device_option(producer_op):
        if producer_op.type == 'CopyGPUToCPU':
            return caffe2_pb2.DeviceOption()
        else:
            return producer_op.device_option
    with ScopedWS('__get_params_from_init_net__', is_reset=True, is_cleanup
        =True) as ws:
        ws.RunNetOnce(init_net)
        params = {b: fetch_any_blob(b) for b in init_net.external_output}
    ssa, versions = core.get_ssa(init_net)
    producer_map = get_producer_map(ssa)
    device_options = {b: _get_device_option(init_net.op[producer_map[b,
        versions[b]][0]]) for b in init_net.external_output}
    return params, device_options
