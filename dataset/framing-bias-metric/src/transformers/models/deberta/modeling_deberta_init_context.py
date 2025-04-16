def init_context(self, reuse_mask=True, scale=1):
    if self.context_stack is None:
        self.context_stack = []
    self.count = 0
    for c in self.context_stack:
        c.reuse_mask = reuse_mask
        c.scale = scale
