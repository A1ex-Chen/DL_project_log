def _init_head(self, ch):
    m = self.model[-1]
    if isinstance(m, Detect):
        s = 256
        m.inplace = self.inplace
        m.stride = torch.tensor([(s / x.shape[-2]) for x in self.forward(
            torch.zeros(1, ch, s, s))])
        m.anchors /= m.stride.view(-1, 1, 1)
        self.stride = m.stride
        self._initialize_biases()
    if isinstance(m, DetectX):
        m.inplace = self.inplace
        self.stride = torch.tensor(m.stride)
        m.initialize_biases()
    if isinstance(m, DetectV8):
        s = 256
        m.inplace = self.inplace
        forward = lambda x: self.forward(x)
        m.stride = torch.tensor([(s / x.shape[-2]) for x in forward(torch.
            zeros(1, ch, s, s))])
        self.stride = m.stride
        m.bias_init()
