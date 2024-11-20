def _measure_backward_torch_strategy(self, operation_outputs):
    helper = BackwardHelper.new_from(operation_outputs)
    backward_ms = self._measure_ms(helper.run_backward)
    accum_grad_ms = self._measure_ms(helper.run_accumulate_grad)
    diff = backward_ms - accum_grad_ms
    return diff if diff >= 1e-06 else backward_ms
