def uncrop_points(points: torch.Tensor, crop_box: List[int]) ->torch.Tensor:
    """Uncrop points by adding the crop box offset."""
    x0, y0, _, _ = crop_box
    offset = torch.tensor([[x0, y0]], device=points.device)
    if len(points.shape) == 3:
        offset = offset.unsqueeze(1)
    return points + offset
