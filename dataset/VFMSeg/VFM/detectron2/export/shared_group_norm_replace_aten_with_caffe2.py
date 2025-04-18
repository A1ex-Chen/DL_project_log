def group_norm_replace_aten_with_caffe2(predict_net: caffe2_pb2.NetDef):
    """
    For ONNX exported model, GroupNorm will be represented as ATen op,
        this can be a drop in replacement from ATen to GroupNorm
    """
    count = 0
    for op in predict_net.op:
        if op.type == 'ATen':
            op_name = get_pb_arg_vals(op, 'operator', None)
            if op_name and op_name.decode() == 'group_norm':
                op.arg.remove(get_pb_arg(op, 'operator'))
                if get_pb_arg_vali(op, 'cudnn_enabled', None):
                    op.arg.remove(get_pb_arg(op, 'cudnn_enabled'))
                num_groups = get_pb_arg_vali(op, 'num_groups', None)
                if num_groups is not None:
                    op.arg.remove(get_pb_arg(op, 'num_groups'))
                    check_set_pb_arg(op, 'group', 'i', num_groups)
                op.type = 'GroupNorm'
                count += 1
    if count > 1:
        logger.info('Replaced {} ATen operator to GroupNormOp'.format(count))
