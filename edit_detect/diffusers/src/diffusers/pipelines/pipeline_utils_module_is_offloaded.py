def module_is_offloaded(module):
    if not is_accelerate_available() or is_accelerate_version('<',
        '0.17.0.dev0'):
        return False
    return hasattr(module, '_hf_hook') and isinstance(module._hf_hook,
        accelerate.hooks.CpuOffload)
