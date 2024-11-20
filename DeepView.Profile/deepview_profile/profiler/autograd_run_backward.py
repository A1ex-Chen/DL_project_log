def run_backward(self):
    for grad_fn in self._grad_fn_ordering:
        outputs = grad_fn(*self._input_holder[grad_fn])
        if isinstance(outputs, torch.Tensor):
            outputs = [outputs]
        for output, (next_fn, input_idx) in zip(outputs, grad_fn.next_functions
            ):
            if next_fn is None or next_fn not in self._input_holder:
                continue
            self._input_holder[next_fn][input_idx] = output
