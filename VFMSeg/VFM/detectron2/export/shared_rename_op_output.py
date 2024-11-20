def rename_op_output(predict_net: caffe2_pb2.NetDef, op_id: int, output_id:
    int, new_name: str):
    """
    Rename the op_id-th operator in predict_net, change it's output_id-th input's
        name to the new_name. It also does automatic re-route and change
        external_output and if necessary.
    - It allows multiple consumers of its output.
    - This function modifies predict_net in-place, doesn't need init_net.
    """
    assert isinstance(predict_net, caffe2_pb2.NetDef)
    ssa, blob_versions = core.get_ssa(predict_net)
    versioned_inputs, versioned_outputs = ssa[op_id]
    old_name, version = versioned_outputs[output_id]
    _rename_versioned_blob_in_proto(predict_net, old_name, new_name,
        version, ssa, {}, blob_versions)
