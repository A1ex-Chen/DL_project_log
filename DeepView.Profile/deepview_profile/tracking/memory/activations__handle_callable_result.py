def _handle_callable_result(self, retval, context):
    if isinstance(retval, torch.Tensor) and retval.grad_fn is not None:
        self.grad_function_contexts[retval.grad_fn] = context
    elif isinstance(retval, tuple) or isinstance(retval, list):
        for inner_value in retval:
            self._handle_callable_result(inner_value, context)
