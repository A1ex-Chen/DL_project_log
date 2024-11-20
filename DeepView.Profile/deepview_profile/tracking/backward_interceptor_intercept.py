@contextlib.contextmanager
def intercept(self):
    self._backward_hooks.attach_hook(torch.Tensor, 'backward', self.
        _hook_creator)
    try:
        yield
    except _SuspendExecution:
        pass
    finally:
        self._backward_hooks.remove_hooks()
