def module_is_sequentially_offloaded(module):
    if not is_accelerate_available() or is_accelerate_version('<', '0.14.0'):
        return False
    return hasattr(module, '_hf_hook') and (isinstance(module._hf_hook,
        accelerate.hooks.AlignDevicesHook) or hasattr(module._hf_hook,
        'hooks') and isinstance(module._hf_hook.hooks[0], accelerate.hooks.
        AlignDevicesHook))
