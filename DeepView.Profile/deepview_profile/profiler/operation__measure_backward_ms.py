def _measure_backward_ms(self, func_name, operation_outputs):
    try:
        return self._measure_backward_engine_strategy(operation_outputs)
    except (RuntimeError, TypeError):
        logger.debug("%s: Falling back to PyTorch's engine", func_name)
        return self._measure_backward_torch_strategy(operation_outputs)
