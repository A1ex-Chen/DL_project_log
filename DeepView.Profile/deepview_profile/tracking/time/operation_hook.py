def hook(*args, **kwargs):
    if self._processing_hook:
        return func(*args, **kwargs)
    self._processing_hook = True
    try:
        stack = CallStack.from_here(self._project_root, start_from=2)
        if len(stack.frames) == 0:
            return func(*args, **kwargs)
        forward_ms, backward_ms = self._profiler.measure_operation_ms(func,
            args, kwargs)
        self.operations.append(OperationInfo(operation_name=func.__name__,
            stack=stack, forward_ms=forward_ms, backward_ms=backward_ms))
        return func(*args, **kwargs)
    finally:
        self._processing_hook = False
