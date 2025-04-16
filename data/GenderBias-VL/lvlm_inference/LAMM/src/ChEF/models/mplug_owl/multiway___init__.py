def __init__(self, module_provider, num_multiway=2, out_features=None):
    super(MultiwayNetwork, self).__init__()
    self.multiway = torch.nn.ModuleList([module_provider() for _ in range(
        num_multiway)])
    self.out_features = out_features
