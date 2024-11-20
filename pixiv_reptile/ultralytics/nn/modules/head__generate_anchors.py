def _generate_anchors(self, shapes, grid_size=0.05, dtype=torch.float32,
    device='cpu', eps=0.01):
    """Generates anchor bounding boxes for given shapes with specific grid size and validates them."""
    anchors = []
    for i, (h, w) in enumerate(shapes):
        sy = torch.arange(end=h, dtype=dtype, device=device)
        sx = torch.arange(end=w, dtype=dtype, device=device)
        grid_y, grid_x = torch.meshgrid(sy, sx, indexing='ij'
            ) if TORCH_1_10 else torch.meshgrid(sy, sx)
        grid_xy = torch.stack([grid_x, grid_y], -1)
        valid_WH = torch.tensor([w, h], dtype=dtype, device=device)
        grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH
        wh = torch.ones_like(grid_xy, dtype=dtype, device=device
            ) * grid_size * 2.0 ** i
        anchors.append(torch.cat([grid_xy, wh], -1).view(-1, h * w, 4))
    anchors = torch.cat(anchors, 1)
    valid_mask = ((anchors > eps) & (anchors < 1 - eps)).all(-1, keepdim=True)
    anchors = torch.log(anchors / (1 - anchors))
    anchors = anchors.masked_fill(~valid_mask, float('inf'))
    return anchors, valid_mask
