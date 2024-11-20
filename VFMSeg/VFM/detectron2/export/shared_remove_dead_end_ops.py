def remove_dead_end_ops(net_def: caffe2_pb2.NetDef):
    """remove ops if its output is not used or not in external_output"""
    ssa, versions = core.get_ssa(net_def)
    versioned_external_output = [(name, versions[name]) for name in net_def
        .external_output]
    consumer_map = get_consumer_map(ssa)
    removed_op_ids = set()

    def _is_dead_end(versioned_blob):
        return not (versioned_blob in versioned_external_output or len(
            consumer_map[versioned_blob]) > 0 and all(x[0] not in
            removed_op_ids for x in consumer_map[versioned_blob]))
    for i, ssa_i in reversed(list(enumerate(ssa))):
        versioned_outputs = ssa_i[1]
        if all(_is_dead_end(outp) for outp in versioned_outputs):
            removed_op_ids.add(i)
    new_ops = [op for i, op in enumerate(net_def.op) if i not in removed_op_ids
        ]
    del net_def.op[:]
    net_def.op.extend(new_ops)
