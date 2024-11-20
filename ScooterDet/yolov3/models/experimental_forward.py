def forward(self, x, augment=False, profile=False, visualize=False):
    y = [module(x, augment, profile, visualize)[0] for module in self]
    y = torch.cat(y, 1)
    return y, None
