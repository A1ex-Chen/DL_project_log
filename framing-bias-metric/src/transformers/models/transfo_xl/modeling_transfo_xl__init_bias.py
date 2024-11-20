def _init_bias(self, bias):
    nn.init.constant_(bias, 0.0)
