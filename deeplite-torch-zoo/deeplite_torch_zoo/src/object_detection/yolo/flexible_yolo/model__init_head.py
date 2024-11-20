def _init_head(self):
    if isinstance(self.detection, Detect):
        s = 256
        self.detection.stride = torch.tensor([(s / x.shape[-2]) for x in
            self.forward(torch.zeros(1, 3, s, s))])
        self.detection.anchors /= self.detection.stride.view(-1, 1, 1)
        self.stride = self.detection.stride
        self._initialize_biases()
    if isinstance(self.detection, DetectV8):
        s = 256
        forward = lambda x: self.forward(x)
        self.detection.stride = torch.tensor([(s / x.shape[-2]) for x in
            forward(torch.zeros(1, 3, s, s))])
        self.stride = self.detection.stride
        self.detection.bias_init()
