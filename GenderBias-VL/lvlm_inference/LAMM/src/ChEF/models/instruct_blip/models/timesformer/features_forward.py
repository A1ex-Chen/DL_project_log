def forward(self, x):
    for name, module in self.items():
        x = module(x)
    out = self.hooks.get_output(x.device)
    return out if self.out_as_dict else list(out.values())
