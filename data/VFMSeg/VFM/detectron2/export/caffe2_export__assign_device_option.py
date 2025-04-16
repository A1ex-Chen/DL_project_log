def _assign_device_option(predict_net: caffe2_pb2.NetDef, init_net:
    caffe2_pb2.NetDef, tensor_inputs: List[torch.Tensor]):
    """
    ONNX exported network doesn't have concept of device, assign necessary
    device option for each op in order to make it runable on GPU runtime.
    """

    def _get_device_type(torch_tensor):
        assert torch_tensor.device.type in ['cpu', 'cuda']
        assert torch_tensor.device.index == 0
        return torch_tensor.device.type

    def _assign_op_device_option(net_proto, net_ssa, blob_device_types):
        for op, ssa_i in zip(net_proto.op, net_ssa):
            if op.type in ['CopyCPUToGPU', 'CopyGPUToCPU']:
                op.device_option.CopyFrom(core.DeviceOption(caffe2_pb2.CUDA, 0)
                    )
            else:
                devices = [blob_device_types[b] for b in ssa_i[0] + ssa_i[1]]
                assert all(d == devices[0] for d in devices)
                if devices[0] == 'cuda':
                    op.device_option.CopyFrom(core.DeviceOption(caffe2_pb2.
                        CUDA, 0))
    predict_net_input_device_types = {(name, 0): _get_device_type(tensor) for
        name, tensor in zip(predict_net.external_input, tensor_inputs)}
    predict_net_device_types = infer_device_type(predict_net, known_status=
        predict_net_input_device_types, device_name_style='pytorch')
    predict_net_ssa, _ = core.get_ssa(predict_net)
    _assign_op_device_option(predict_net, predict_net_ssa,
        predict_net_device_types)
    init_net_ssa, versions = core.get_ssa(init_net)
    init_net_output_device_types = {(name, versions[name]):
        predict_net_device_types[name, 0] for name in init_net.external_output}
    init_net_device_types = infer_device_type(init_net, known_status=
        init_net_output_device_types, device_name_style='pytorch')
    _assign_op_device_option(init_net, init_net_ssa, init_net_device_types)
