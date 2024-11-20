def _callable_hook_creator(self, func):

    def hook(*args, **kwargs):
        if self._processing_hook:
            return func(*args, **kwargs)
        self._processing_hook = True
        try:
            retval = func(*args, **kwargs)
        finally:
            self._processing_hook = False
        if not isinstance(retval, torch.Tensor) and not isinstance(retval,
            tuple) and not isinstance(retval, list):
            return retval
        if isinstance(retval, torch.Tensor) and (not retval.is_cuda or 
            retval.grad_fn is None):
            return retval
        stack = CallStack.from_here(self._project_root, start_from=2)
        if len(stack.frames) == 0:
            return retval
        context = OperationContext(operation_name=func.__name__, stack=stack)
        self._handle_callable_result(retval, context)
        return retval
    return hook
