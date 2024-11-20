def _extract_relevant_gradient_functions(self, model_output,
    grad_function_contexts):
    gradient_functions = _extract_gradient_functions_in_topological_order(
        model_output)
    relevant_grad_fns = []
    relevant_contexts = {}
    for grad_fn in gradient_functions:
        if grad_fn not in grad_function_contexts:
            continue
        relevant_grad_fns.append(grad_fn)
        relevant_contexts[grad_fn] = grad_function_contexts[grad_fn]
    return relevant_grad_fns, relevant_contexts
