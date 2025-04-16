def forward(self, input):
    for module in self._modules.values():
        input = module(input)
    return input
