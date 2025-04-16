@staticmethod
def _get_tracing_name(module, tracing_state):
    if not tracing_state._traced_module_stack:
        return None
    parent_module = tracing_state._traced_module_stack[-1]
    for name, child in parent_module.named_children():
        if child is module:
            return name
    return None
