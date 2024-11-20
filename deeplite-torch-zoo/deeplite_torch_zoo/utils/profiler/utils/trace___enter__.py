def __enter__(self):
    self.original_slow_forward = torch.nn.Module._slow_forward
    torch.nn.Module._slow_forward = self._patched_slow_forward
