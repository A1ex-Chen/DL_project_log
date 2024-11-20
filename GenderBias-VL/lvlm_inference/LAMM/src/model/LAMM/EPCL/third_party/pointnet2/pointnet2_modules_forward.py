def forward(self, xyz2: torch.Tensor, xyz1: torch.Tensor, features2: torch.
    Tensor, features1: torch.Tensor) ->torch.Tensor:
    """Propagate features from xyz1 to xyz2.
        Parameters
        ----------
        xyz2 : torch.Tensor
            (B, N2, 3) tensor of the xyz coordinates of the features
        xyz1 : torch.Tensor
            (B, N1, 3) tensor of the xyz coordinates of the features
        features2 : torch.Tensor
            (B, C2, N2) tensor of the descriptors of the the features
        features1 : torch.Tensor
            (B, C1, N1) tensor of the descriptors of the the features

        Returns
        -------
        new_features1 : torch.Tensor
            (B, \\sum_k(mlps[k][-1]), N1) tensor of the new_features descriptors
        """
    new_features_list = []
    for i in range(len(self.groupers)):
        new_features = self.groupers[i](xyz1, xyz2, features1)
        new_features = self.mlps[i](new_features)
        new_features = F.max_pool2d(new_features, kernel_size=[1,
            new_features.size(3)])
        new_features = new_features.squeeze(-1)
        if features2 is not None:
            new_features = torch.cat([new_features, features2], dim=1)
        new_features = new_features.unsqueeze(-1)
        new_features = self.post_mlp(new_features)
        new_features_list.append(new_features)
    return torch.cat(new_features_list, dim=1).squeeze(-1)
