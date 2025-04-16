def _update_i(op, ssa_i):
    versioned_inputs = ssa_i[0]
    versioned_outputs = ssa_i[1]
    inputs_status = [_known_status.get(b, None) for b in versioned_inputs]
    outputs_status = [_known_status.get(b, None) for b in versioned_outputs]
    new_inputs_status, new_outputs_status = status_updater(op,
        inputs_status, outputs_status)
    for versioned_blob, status in zip(versioned_inputs + versioned_outputs,
        new_inputs_status + new_outputs_status):
        if status is not None:
            _check_and_update(versioned_blob, status)
