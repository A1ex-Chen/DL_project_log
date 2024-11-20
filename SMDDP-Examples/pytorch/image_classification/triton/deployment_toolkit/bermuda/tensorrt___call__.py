def __call__(self, x):
    buffers = self._prepare_buffers_if_needed(x)
    bindings = self._update_bindings(buffers)
    for name in self._input_names:
        cuda.memcpy_htod(buffers.x_dev[name], buffers.x_host[name])
    self._cuda_context.push()
    self._context.execute_v2(bindings=bindings)
    self._cuda_context.pop()
    for name in self._output_names:
        cuda.memcpy_dtoh(buffers.y_pred_host[name], buffers.y_pred_dev[name])
    return buffers.y_pred_host
