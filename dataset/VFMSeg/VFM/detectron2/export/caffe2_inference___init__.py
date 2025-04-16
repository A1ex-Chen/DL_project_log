def __init__(self, predict_net, init_net, *, convert_outputs=None):
    """
        Args:
            predict_net, init_net (core.Net): caffe2 nets
            convert_outptus (callable): a function that converts caffe2
                outputs to the same format of the original pytorch model.
                By default, use the one defined in the caffe2 meta_arch.
        """
    super().__init__()
    self.protobuf_model = ProtobufModel(predict_net, init_net)
    self.size_divisibility = get_pb_arg_vali(predict_net,
        'size_divisibility', 0)
    self.device = get_pb_arg_vals(predict_net, 'device', b'cpu').decode('ascii'
        )
    if convert_outputs is None:
        meta_arch = get_pb_arg_vals(predict_net, 'meta_architecture',
            b'GeneralizedRCNN')
        meta_arch = META_ARCH_CAFFE2_EXPORT_TYPE_MAP[meta_arch.decode('ascii')]
        self._convert_outputs = meta_arch.get_outputs_converter(predict_net,
            init_net)
    else:
        self._convert_outputs = convert_outputs
