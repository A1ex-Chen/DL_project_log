def _fuse_once(predict_net):
    ssa, blob_versions = core.get_ssa(predict_net)
    consumer_map = get_consumer_map(ssa)
    versioned_external_output = [(name, blob_versions[name]) for name in
        predict_net.external_output]
    for op_id, op in enumerate(predict_net.op):
        if op.type in _COPY_OPS:
            fw_copy_versioned_output = ssa[op_id][1][0]
            consumer_ids = [x[0] for x in consumer_map[
                fw_copy_versioned_output]]
            reverse_op_type = _COPY_OPS[1 - _COPY_OPS.index(op.type)]
            is_fusable = (len(consumer_ids) > 0 and 
                fw_copy_versioned_output not in versioned_external_output and
                all(predict_net.op[_op_id].type == reverse_op_type and ssa[
                _op_id][1][0] not in versioned_external_output for _op_id in
                consumer_ids))
            if is_fusable:
                for rv_copy_op_id in consumer_ids:
                    rs_copy_versioned_output = ssa[rv_copy_op_id][1][0]
                    next_op_id, inp_id = consumer_map[rs_copy_versioned_output
                        ][0]
                    predict_net.op[next_op_id].input[inp_id] = op.input[0]
                new_ops = [op for i, op in enumerate(predict_net.op) if i !=
                    op_id and i not in consumer_ids]
                del predict_net.op[:]
                predict_net.op.extend(new_ops)
                return True
    return False
