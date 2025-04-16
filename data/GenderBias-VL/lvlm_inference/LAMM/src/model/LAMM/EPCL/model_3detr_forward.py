def forward(self, xyz: torch.Tensor, features: torch.Tensor=None, inds:
    torch.Tensor=None) ->(torch.Tensor, torch.Tensor):
    for i in range(len(self.layers)):
        if features is not None:
            features = features.type(torch.float32)
        xyz, features, inds = self.layers[i](xyz, features)
    return xyz, features, inds
