def apply_fn(module):
    if hasattr(module, name):
        getattr(module, name).requires_grad_(False)
