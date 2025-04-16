def forward(self, x, augment=False, profile=False, visualize=False):
    """Function generates the YOLO network's final layer."""
    y = [module(x, augment, profile, visualize)[0] for module in self]
    y = torch.cat(y, 2)
    return y, None
