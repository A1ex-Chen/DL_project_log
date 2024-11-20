def _pe_encoding(self, coords: torch.Tensor) ->torch.Tensor:
    """Positionally encode points that are normalized to [0,1]."""
    coords = 2 * coords - 1
    coords = coords @ self.positional_encoding_gaussian_matrix
    coords = 2 * np.pi * coords
    return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)
