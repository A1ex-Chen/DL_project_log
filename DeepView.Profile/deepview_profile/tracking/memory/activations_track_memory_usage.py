def track_memory_usage(self, iteration, input_provider, user_code_path):
    model_output, grad_function_contexts = self._get_grad_function_contexts(
        iteration, input_provider, user_code_path)
    gradient_functions_topo_order, grad_function_contexts = (self.
        _extract_relevant_gradient_functions(model_output,
        grad_function_contexts))
    gradient_functions_topo_order.reverse()
    del model_output
    gc.collect()
    while len(gradient_functions_topo_order) > 0:
        grad_fn = gradient_functions_topo_order.pop()
        context = grad_function_contexts[grad_fn]
        del grad_function_contexts[grad_fn]
        mem_before = torch.cuda.memory_allocated()
        del grad_fn
        gc.collect()
        mem_after = torch.cuda.memory_allocated()
        delta = mem_after - mem_before
        self._activations.append(ActivationEntry(*context, size_bytes=-delta))
