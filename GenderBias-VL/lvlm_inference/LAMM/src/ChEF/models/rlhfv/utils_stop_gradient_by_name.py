def stop_gradient_by_name(name: str):

    def apply_fn(module):
        if hasattr(module, name):
            getattr(module, name).requires_grad_(False)
    return apply_fn
