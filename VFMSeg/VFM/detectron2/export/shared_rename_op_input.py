def rename_op_input(predict_net: caffe2_pb2.NetDef, init_net: caffe2_pb2.
    NetDef, op_id: int, input_id: int, new_name: str, from_producer: bool=False
    ):
    """
    Rename the op_id-th operator in predict_net, change it's input_id-th input's
        name to the new_name. It also does automatic re-route and change
        external_input and init_net if necessary.
    - It requires the input is only consumed by this op.
    - This function modifies predict_net and init_net in-place.
    - When from_producer is enable, this also updates other operators that consumes
        the same input. Be cautious because may trigger unintended behavior.
    """
    assert isinstance(predict_net, caffe2_pb2.NetDef)
    assert isinstance(init_net, caffe2_pb2.NetDef)
    init_net_ssa, init_net_versions = core.get_ssa(init_net)
    predict_net_ssa, predict_net_versions = core.get_ssa(predict_net, copy.
        deepcopy(init_net_versions))
    versioned_inputs, versioned_outputs = predict_net_ssa[op_id]
    old_name, version = versioned_inputs[input_id]
    if from_producer:
        producer_map = get_producer_map(predict_net_ssa)
        if not (old_name, version) in producer_map:
            raise NotImplementedError(
                "Can't find producer, the input {} is probably from init_net, this is not supported yet."
                .format(old_name))
        producer = producer_map[old_name, version]
        rename_op_output(predict_net, producer[0], producer[1], new_name)
        return

    def contain_targets(op_ssa):
        return (old_name, version) in op_ssa[0]
    is_consumer = [contain_targets(op_ssa) for op_ssa in predict_net_ssa]
    if sum(is_consumer) > 1:
        raise IllegalGraphTransformError((
            "Input '{}' of operator(#{}) are consumed by other ops, please use"
             +
            """ rename_op_output on the producer instead. Offending op: 
{}"""
            ).format(old_name, op_id, predict_net.op[op_id]))
    _rename_versioned_blob_in_proto(init_net, old_name, new_name, version,
        init_net_ssa, {}, init_net_versions)
    _rename_versioned_blob_in_proto(predict_net, old_name, new_name,
        version, predict_net_ssa, init_net_versions, predict_net_versions)
