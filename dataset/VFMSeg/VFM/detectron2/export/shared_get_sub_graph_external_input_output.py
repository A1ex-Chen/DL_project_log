def get_sub_graph_external_input_output(predict_net: caffe2_pb2.NetDef,
    sub_graph_op_indices: List[int]) ->Tuple[List[Tuple[str, int]], List[
    Tuple[str, int]]]:
    """
    Return the list of external input/output of sub-graph,
    each element is tuple of the name and corresponding version in predict_net.

    external input/output is defined the same way as caffe2 NetDef.
    """
    ssa, versions = core.get_ssa(predict_net)
    all_inputs = []
    all_outputs = []
    for op_id in sub_graph_op_indices:
        all_inputs += [inp for inp in ssa[op_id][0] if inp not in all_inputs]
        all_outputs += list(ssa[op_id][1])
    ext_inputs = [inp for inp in all_inputs if inp not in all_outputs]
    all_other_inputs = sum((ssa[i][0] for i in range(len(ssa)) if i not in
        sub_graph_op_indices), [(outp, versions[outp]) for outp in
        predict_net.external_output])
    ext_outputs = [outp for outp in all_outputs if outp in set(
        all_other_inputs)]
    return ext_inputs, ext_outputs
