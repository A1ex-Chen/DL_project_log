def forward(self, xyz, new_xyz, features=None):
    """
        Parameters
        ----------
        xyz : torch.Tensor
            xyz coordinates of the features (B, N, 3)
        new_xyz : torch.Tensor
            Ignored
        features : torch.Tensor
            Descriptors of the features (B, C, N)

        Returns
        -------
        new_features : torch.Tensor
            (B, C + 3, 1, N) tensor
        """
    grouped_xyz = xyz.transpose(1, 2).unsqueeze(2)
    if features is not None:
        grouped_features = features.unsqueeze(2)
        if self.use_xyz:
            new_features = torch.cat([grouped_xyz, grouped_features], dim=1)
        else:
            new_features = grouped_features
    else:
        new_features = grouped_xyz
    if self.ret_grouped_xyz:
        return new_features, grouped_xyz
    else:
        return new_features
