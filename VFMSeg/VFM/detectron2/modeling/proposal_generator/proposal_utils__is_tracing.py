def _is_tracing():
    if torch.jit.is_scripting():
        return False
    else:
        return torch.jit.is_tracing()
