def set_fg2cls(self, fg2cls):
    assert self.local_config.use_pointer
    self.register_buffer('fg2cls', nn.Parameter(fg2cls, requires_grad=False))
