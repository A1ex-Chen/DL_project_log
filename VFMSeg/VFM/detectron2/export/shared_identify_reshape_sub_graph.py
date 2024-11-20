def identify_reshape_sub_graph(predict_net: caffe2_pb2.NetDef) ->List[List[int]
    ]:
    """
    Idenfity the reshape sub-graph in a protobuf.
    The reshape sub-graph is defined as matching the following pattern:

    (input_blob) -> Op_1 -> ... -> Op_N -> (new_shape) -─┐
        └-------------------------------------------> Reshape -> (output_blob)

    Return:
        List of sub-graphs, each sub-graph is represented as a list of indices
        of the relavent ops, [Op_1, Op_2, ..., Op_N, Reshape]
    """
    ssa, _ = core.get_ssa(predict_net)
    ret = []
    for i, op in enumerate(predict_net.op):
        if op.type == 'Reshape':
            assert len(op.input) == 2
            input_ssa = ssa[i][0]
            data_source = input_ssa[0]
            shape_source = input_ssa[1]
            op_indices = _get_dependency_chain(ssa, shape_source, data_source)
            ret.append(op_indices + [i])
    return ret
