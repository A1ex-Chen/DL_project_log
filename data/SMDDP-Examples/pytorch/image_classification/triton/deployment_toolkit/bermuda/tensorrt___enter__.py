def __enter__(self):
    self._context = self._engine.create_execution_context()
    self._context.__enter__()
    self._input_names = [self._engine[idx] for idx in range(self._engine.
        num_bindings) if self._engine.binding_is_input(idx)]
    self._output_names = [self._engine[idx] for idx in range(self._engine.
        num_bindings) if not self._engine.binding_is_input(idx)]
    self._has_dynamic_shapes = not self._context.all_binding_shapes_specified
    return self
