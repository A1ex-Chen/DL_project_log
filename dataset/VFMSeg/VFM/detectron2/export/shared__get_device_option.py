def _get_device_option(producer_op):
    if producer_op.type == 'CopyGPUToCPU':
        return caffe2_pb2.DeviceOption()
    else:
        return producer_op.device_option
