def fuse_alias_placeholder(predict_net, init_net):
    """Remove AliasWithName placeholder and rename the input/output of it"""
    for i, op in enumerate(predict_net.op):
        if op.type == 'AliasWithName':
            assert len(op.input) == 1
            assert len(op.output) == 1
            name = get_pb_arg_vals(op, 'name', None).decode()
            is_backward = bool(get_pb_arg_vali(op, 'is_backward', 0))
            rename_op_input(predict_net, init_net, i, 0, name,
                from_producer=is_backward)
            rename_op_output(predict_net, i, 0, name)
    new_ops = []
    for op in predict_net.op:
        if op.type != 'AliasWithName':
            new_ops.append(op)
        else:
            assert op.input == op.output
            assert op.input[0] == op.arg[0].s.decode()
    del predict_net.op[:]
    predict_net.op.extend(new_ops)
