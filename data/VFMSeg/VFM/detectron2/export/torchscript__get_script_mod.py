def _get_script_mod(mod):
    if isinstance(mod, torch.jit.TracedModule):
        return mod._actual_script_module
    return mod
