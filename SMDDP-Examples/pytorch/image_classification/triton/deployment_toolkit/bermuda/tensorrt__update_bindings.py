def _update_bindings(self, buffers: TRTBuffers):
    bindings = [None] * self._engine.num_bindings
    for name in buffers.y_pred_dev:
        binding_idx: int = self._engine[name]
        bindings[binding_idx] = buffers.y_pred_dev[name]
    for name in buffers.x_dev:
        binding_idx: int = self._engine[name]
        bindings[binding_idx] = buffers.x_dev[name]
    return bindings
