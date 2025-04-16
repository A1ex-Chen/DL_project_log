def _measure_backward_engine_strategy(self, operation_outputs):
    engine = AutogradEngine.new_from(operation_outputs)
    return self._measure_ms(engine.run_backward)
