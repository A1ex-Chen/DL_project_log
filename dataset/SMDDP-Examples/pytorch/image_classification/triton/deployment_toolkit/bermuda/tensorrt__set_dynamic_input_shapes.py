def _set_dynamic_input_shapes(self, x_host):

    def _is_shape_dynamic(input_shape):
        return any([(dim is None or dim == -1) for dim in input_shape])
    for name in self._input_names:
        bindings_idx = self._engine[name]
        data_shape = x_host[name].shape
        if self._engine.is_shape_binding(bindings_idx):
            input_shape = self._context.get_shape(bindings_idx)
            if _is_shape_dynamic(input_shape):
                self._context.set_shape_input(bindings_idx, data_shape)
        else:
            input_shape = self._engine.get_binding_shape(bindings_idx)
            if _is_shape_dynamic(input_shape):
                self._context.set_binding_shape(bindings_idx, data_shape)
    assert self._context.all_binding_shapes_specified and self._context.all_shape_inputs_specified
