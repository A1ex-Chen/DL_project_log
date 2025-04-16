def maybe_autocast(self, dtype=torch.float16):
    enable_autocast = self.device != torch.device('cpu')
    if enable_autocast:
        return torch.cuda.amp.autocast(dtype=dtype)
    else:
        return contextlib.nullcontext()
