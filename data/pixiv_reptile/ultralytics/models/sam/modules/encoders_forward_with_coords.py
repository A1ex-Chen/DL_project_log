def forward_with_coords(self, coords_input: torch.Tensor, image_size: Tuple
    [int, int]) ->torch.Tensor:
    """Positionally encode points that are not normalized to [0,1]."""
    coords = coords_input.clone()
    coords[:, :, 0] = coords[:, :, 0] / image_size[1]
    coords[:, :, 1] = coords[:, :, 1] / image_size[0]
    return self._pe_encoding(coords.to(torch.float))
