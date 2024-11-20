def __init__(self, srcs, hdrs, deps, copts, name=None):
    super().__init__(name)
    self.srcs = srcs
    self.hdrs = hdrs
    self.deps = deps
    self.copts = copts
