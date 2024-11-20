def _assign_op_device_option(net_proto, net_ssa, blob_device_types):
    for op, ssa_i in zip(net_proto.op, net_ssa):
        if op.type in ['CopyCPUToGPU', 'CopyGPUToCPU']:
            op.device_option.CopyFrom(core.DeviceOption(caffe2_pb2.CUDA, 0))
        else:
            devices = [blob_device_types[b] for b in ssa_i[0] + ssa_i[1]]
            assert all(d == devices[0] for d in devices)
            if devices[0] == 'cuda':
                op.device_option.CopyFrom(core.DeviceOption(caffe2_pb2.CUDA, 0)
                    )
