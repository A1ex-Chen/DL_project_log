def _get_grad_function_contexts(self, iteration, input_provider, user_code_path
    ):
    grad_function_tracker = GradFunctionTracker(self._project_root)
    backward_interceptor = BackwardInterceptor()
    with grad_function_tracker.track(), backward_interceptor.intercept(
        ), user_code_environment(user_code_path, self._project_root):
        iteration(*input_provider())
    return (backward_interceptor.backward_root, grad_function_tracker.
        grad_function_contexts)
