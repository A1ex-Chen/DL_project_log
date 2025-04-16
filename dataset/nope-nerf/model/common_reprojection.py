def reprojection(pixels, depth, Rt_ref, world_mat, camera_mat):
    assert pixels.shape[-1] == 2
    pixels, is_numpy = to_pytorch(pixels, True)
    depth = to_pytorch(depth)
    camera_mat = to_pytorch(camera_mat)
    Rt_ref = to_pytorch(Rt_ref)
    pixels = pixels.permute(0, 2, 1)
    pixels = torch.cat([pixels, torch.ones_like(pixels)], dim=1)
    pixels_depth = pixels.clone()
    depth = depth.view(1, -1, 1)
    pixels_depth[:, :3] = pixels[:, :3] * depth.permute(0, 2, 1)
    xy_ref = camera_mat @ Rt_ref @ torch.inverse(world_mat) @ torch.inverse(
        camera_mat) @ pixels_depth
    xy_ref = xy_ref[:, :3].permute(0, 2, 1)
    xy_ref = xy_ref[..., :2] / xy_ref[..., 2:]
    valid_points = xy_ref.abs().max(dim=-1)[0] <= 1
    valid_mask = valid_points.unsqueeze(-1).float()
    if is_numpy:
        xy_ref = xy_ref.numpy()
    return xy_ref, valid_mask
