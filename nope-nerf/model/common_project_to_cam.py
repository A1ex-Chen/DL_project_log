def project_to_cam(points, camera_mat, device):
    """
    points: (B, N, 3)
    camera_mat: (B, 4, 4)
    """
    B, N, D = points.size()
    points, is_numpy = to_pytorch(points, True)
    points = points.permute(0, 2, 1)
    points = torch.cat([points, torch.ones(B, 1, N, device=device)], dim=1)
    xy_ref = camera_mat @ points
    xy_ref = xy_ref[:, :3].permute(0, 2, 1)
    xy_ref = xy_ref[..., :2] / xy_ref[..., 2:]
    valid_points = xy_ref.abs().max(dim=-1)[0] <= 1
    valid_mask = valid_points.unsqueeze(-1).bool()
    if is_numpy:
        xy_ref = xy_ref.numpy()
    return xy_ref, valid_mask
