def remove_reshape_for_fc(predict_net, params):
    """
    In PyTorch nn.Linear has to take 2D tensor, this often leads to reshape
        a 4D tensor to 2D by calling .view(). However this (dynamic) reshaping
        doesn't work well with ONNX and Int8 tools, and cause using extra
        ops (eg. ExpandDims) that might not be available on mobile.
    Luckily Caffe2 supports 4D tensor for FC, so we can remove those reshape
        after exporting ONNX model.
    """
    from caffe2.python import core
    reshape_sub_graphs = identify_reshape_sub_graph(predict_net)
    sub_graphs_to_remove = []
    for reshape_sub_graph in reshape_sub_graphs:
        reshape_op_id = reshape_sub_graph[-1]
        assert predict_net.op[reshape_op_id].type == 'Reshape'
        ssa, _ = core.get_ssa(predict_net)
        reshape_output = ssa[reshape_op_id][1][0]
        consumers = [i for i in range(len(ssa)) if reshape_output in ssa[i][0]]
        if all(predict_net.op[consumer].type == 'FC' for consumer in consumers
            ):
            ext_inputs, ext_outputs = get_sub_graph_external_input_output(
                predict_net, reshape_sub_graph)
            non_params_ext_inputs = [inp for inp in ext_inputs if inp[1] != 0]
            if len(non_params_ext_inputs) == 1 and len(ext_outputs) == 1:
                sub_graphs_to_remove.append(reshape_sub_graph)
    remove_op_ids = []
    params_to_remove = []
    for sub_graph in sub_graphs_to_remove:
        logger.info('Remove Reshape sub-graph:\n{}'.format(''.join([
            '(#{:>4})\n{}'.format(i, predict_net.op[i]) for i in sub_graph])))
        reshape_op_id = sub_graph[-1]
        new_reshap_output = predict_net.op[reshape_op_id].input[0]
        rename_op_output(predict_net, reshape_op_id, 0, new_reshap_output)
        ext_inputs, ext_outputs = get_sub_graph_external_input_output(
            predict_net, sub_graph)
        non_params_ext_inputs = [inp for inp in ext_inputs if inp[1] != 0]
        params_ext_inputs = [inp for inp in ext_inputs if inp[1] == 0]
        assert len(non_params_ext_inputs) == 1 and len(ext_outputs) == 1
        assert ext_outputs[0][0] == non_params_ext_inputs[0][0]
        assert ext_outputs[0][1] == non_params_ext_inputs[0][1] + 1
        remove_op_ids.extend(sub_graph)
        params_to_remove.extend(params_ext_inputs)
    predict_net = copy.deepcopy(predict_net)
    new_ops = [op for i, op in enumerate(predict_net.op) if i not in
        remove_op_ids]
    del predict_net.op[:]
    predict_net.op.extend(new_ops)
    for versioned_params in params_to_remove:
        name = versioned_params[0]
        logger.info(
            'Remove params: {} from init_net and predict_net.external_input'
            .format(name))
        del params[name]
        predict_net.external_input.remove(name)
    return predict_net, params
