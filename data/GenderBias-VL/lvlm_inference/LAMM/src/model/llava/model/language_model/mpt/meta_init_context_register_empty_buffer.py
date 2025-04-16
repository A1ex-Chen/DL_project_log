def register_empty_buffer(module, name, buffer):
    old_register_buffer(module, name, buffer)
    if buffer is not None:
        module._buffers[name] = module._buffers[name].to(device)
