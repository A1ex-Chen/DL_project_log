def _prepare_buffers_if_needed(self, x_host: Dict[str, object]):
    new_batch_size = list(x_host.values())[0].shape[0]
    current_batch_size = list(self._buffers.y_pred_host.values())[0].shape[0
        ] if self._buffers else 0
    if self._has_dynamic_shapes or new_batch_size != current_batch_size:
        self._set_dynamic_input_shapes(x_host)
        y_pred_host = {}
        for name in self._output_names:
            shape = self._context.get_binding_shape(self._engine[name])
            y_pred_host[name] = np.zeros(shape, dtype=trt.nptype(self.
                _model.outputs[name].dtype))
        y_pred_dev = {name: cuda.mem_alloc(data.nbytes) for name, data in
            y_pred_host.items()}
        x_dev = {name: cuda.mem_alloc(host_input.nbytes) for name,
            host_input in x_host.items() if name in self._input_names}
        self._buffers = TRTBuffers(None, x_dev, y_pred_host, y_pred_dev)
    return self._buffers._replace(x_host=x_host)
