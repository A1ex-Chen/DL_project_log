def _generic_status_identifier(predict_net: caffe2_pb2.NetDef,
    status_updater: Callable, known_status: Dict[Tuple[str, int], Any]) ->Dict[
    Tuple[str, int], Any]:
    """
    Statically infer the status of each blob, the status can be such as device type
        (CPU/GPU), layout (NCHW/NHWC), data type (float32/int8), etc. "Blob" here
        is versioned blob (Tuple[str, int]) in the format compatible with ssa.
    Inputs:
        predict_net: the caffe2 network
        status_updater: a callable, given an op and the status of its input/output,
            it returns the updated status of input/output. `None` is used for
            representing unknown status.
        known_status: a dict containing known status, used as initialization.
    Outputs:
        A dict mapping from versioned blob to its status
    """
    ssa, versions = core.get_ssa(predict_net)
    versioned_ext_input = [(b, 0) for b in predict_net.external_input]
    versioned_ext_output = [(b, versions[b]) for b in predict_net.
        external_output]
    all_versioned_blobs = set().union(*[set(x[0] + x[1]) for x in ssa])
    allowed_vbs = all_versioned_blobs.union(versioned_ext_input).union(
        versioned_ext_output)
    assert all(k in allowed_vbs for k in known_status)
    assert all(v is not None for v in known_status.values())
    _known_status = copy.deepcopy(known_status)

    def _check_and_update(key, value):
        assert value is not None
        if key in _known_status:
            if not _known_status[key] == value:
                raise RuntimeError(
                    'Confilict status for {}, existing status {}, new status {}'
                    .format(key, _known_status[key], value))
        _known_status[key] = value

    def _update_i(op, ssa_i):
        versioned_inputs = ssa_i[0]
        versioned_outputs = ssa_i[1]
        inputs_status = [_known_status.get(b, None) for b in versioned_inputs]
        outputs_status = [_known_status.get(b, None) for b in versioned_outputs
            ]
        new_inputs_status, new_outputs_status = status_updater(op,
            inputs_status, outputs_status)
        for versioned_blob, status in zip(versioned_inputs +
            versioned_outputs, new_inputs_status + new_outputs_status):
            if status is not None:
                _check_and_update(versioned_blob, status)
    for op, ssa_i in zip(predict_net.op, ssa):
        _update_i(op, ssa_i)
    for op, ssa_i in zip(reversed(predict_net.op), reversed(ssa)):
        _update_i(op, ssa_i)
    for k in all_versioned_blobs:
        if k not in _known_status:
            raise NotImplementedError(
                'Can not infer the status for {}. Currently only support the case where a single forward and backward pass can identify status for all blobs.'
                .format(k))
    return _known_status
