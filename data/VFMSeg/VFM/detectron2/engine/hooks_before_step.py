def before_step(self):
    if self._enable_predicate(self.trainer):
        self._profiler = torch.autograd.profiler.profile(use_cuda=self.
            _use_cuda)
        self._profiler.__enter__()
    else:
        self._profiler = None
