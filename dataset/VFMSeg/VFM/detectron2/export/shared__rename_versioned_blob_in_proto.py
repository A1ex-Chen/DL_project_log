def _rename_versioned_blob_in_proto(proto: caffe2_pb2.NetDef, old_name: str,
    new_name: str, version: int, ssa: List[Tuple[List[Tuple[str, int]],
    List[Tuple[str, int]]]], start_versions: Dict[str, int], end_versions:
    Dict[str, int]):
    """In given proto, rename all blobs with matched version"""
    for op, i_th_ssa in zip(proto.op, ssa):
        versioned_inputs, versioned_outputs = i_th_ssa
        for i in range(len(op.input)):
            if versioned_inputs[i] == (old_name, version):
                op.input[i] = new_name
        for i in range(len(op.output)):
            if versioned_outputs[i] == (old_name, version):
                op.output[i] = new_name
    if start_versions.get(old_name, 0) == version:
        for i in range(len(proto.external_input)):
            if proto.external_input[i] == old_name:
                proto.external_input[i] = new_name
    if end_versions.get(old_name, 0) == version:
        for i in range(len(proto.external_output)):
            if proto.external_output[i] == old_name:
                proto.external_output[i] = new_name
