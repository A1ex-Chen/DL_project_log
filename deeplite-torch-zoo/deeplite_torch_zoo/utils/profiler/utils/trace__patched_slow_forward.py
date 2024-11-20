@staticmethod
def _patched_slow_forward(module, *inputs, **kwargs):
    tracing_state = torch._C._get_tracing_state()
    if not tracing_state or isinstance(module.forward, torch._C.ScriptMethod):
        return module.forward(*inputs, **kwargs)
    if not hasattr(tracing_state, '_traced_module_stack'):
        tracing_state._traced_module_stack = []
    module_name = ScopeNameContextManager._get_tracing_name(module,
        tracing_state)
    scope_name = (f'{module._get_name()}[{module_name}]' if module_name else
        module._get_name())
    tracing_state.push_scope(scope_name)
    tracing_state._traced_module_stack.append(module)
    try:
        result = module.forward(*inputs, **kwargs)
    finally:
        tracing_state.pop_scope()
        tracing_state._traced_module_stack.pop()
    return result
