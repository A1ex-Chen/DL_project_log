def export_caffe2_detection_model(model: torch.nn.Module, tensor_inputs:
    List[torch.Tensor]):
    """
    Export a caffe2-compatible Detectron2 model to caffe2 format via ONNX.

    Arg:
        model: a caffe2-compatible version of detectron2 model, defined in caffe2_modeling.py
        tensor_inputs: a list of tensors that caffe2 model takes as input.
    """
    model = copy.deepcopy(model)
    assert isinstance(model, torch.nn.Module)
    assert hasattr(model, 'encode_additional_info')
    logger.info('Exporting a {} model via ONNX ...'.format(type(model).
        __name__) +
        ' Some warnings from ONNX are expected and are usually not to worry about.'
        )
    onnx_model = export_onnx_model(model, (tensor_inputs,))
    init_net, predict_net = Caffe2Backend.onnx_graph_to_caffe2_net(onnx_model)
    ops_table = [[op.type, op.input, op.output] for op in predict_net.op]
    table = tabulate(ops_table, headers=['type', 'input', 'output'],
        tablefmt='pipe')
    logger.info(
        'ONNX export Done. Exported predict_net (before optimizations):\n' +
        colored(table, 'cyan'))
    fuse_alias_placeholder(predict_net, init_net)
    if any(t.device.type != 'cpu' for t in tensor_inputs):
        fuse_copy_between_cpu_and_gpu(predict_net)
        remove_dead_end_ops(init_net)
        _assign_device_option(predict_net, init_net, tensor_inputs)
    params, device_options = get_params_from_init_net(init_net)
    predict_net, params = remove_reshape_for_fc(predict_net, params)
    init_net = construct_init_net_from_params(params, device_options)
    group_norm_replace_aten_with_caffe2(predict_net)
    model.encode_additional_info(predict_net, init_net)
    logger.info('Operators used in predict_net: \n{}'.format(_op_stats(
        predict_net)))
    logger.info('Operators used in init_net: \n{}'.format(_op_stats(init_net)))
    return predict_net, init_net
