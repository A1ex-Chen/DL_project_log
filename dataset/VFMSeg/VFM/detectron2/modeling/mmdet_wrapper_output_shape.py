def output_shape(self) ->Dict[str, ShapeSpec]:
    return {k: v for k, v in zip(self._output_names, self._output_shapes)}
