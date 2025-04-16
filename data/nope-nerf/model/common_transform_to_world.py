def transform_to_world(pixels, depth, camera_mat, world_mat=None, scale_mat
    =None, invert=True, device=torch.device('cuda')):
    """ Transforms pixel positions p with given depth value d to world coordinates.

    Args:
        pixels (tensor): pixel tensor of size B x N x 2
        depth (tensor): depth tensor of size B x N x 1
        camera_mat (tensor): camera matrix
        world_mat (tensor): world matrix
        scale_mat (tensor): scale matrix
        invert (bool): whether to invert matrices (default: true)
    """
    assert pixels.shape[-1] == 2
    if world_mat is None:
        world_mat = torch.tensor([[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],
            [0, 0, 0, 1]]], dtype=torch.float32, device=device)
    if scale_mat is None:
        scale_mat = torch.tensor([[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],
            [0, 0, 0, 1]]], dtype=torch.float32, device=device)
    pixels, is_numpy = to_pytorch(pixels, True)
    depth = to_pytorch(depth)
    camera_mat = to_pytorch(camera_mat)
    world_mat = to_pytorch(world_mat)
    scale_mat = to_pytorch(scale_mat)
    if invert:
        camera_mat = torch.inverse(camera_mat)
        world_mat = torch.inverse(world_mat)
        scale_mat = torch.inverse(scale_mat)
    pixels = pixels.permute(0, 2, 1)
    pixels = torch.cat([pixels, torch.ones_like(pixels)], dim=1)
    pixels_depth = pixels.clone()
    pixels_depth[:, :3] = pixels[:, :3] * depth.permute(0, 2, 1)
    p_world = scale_mat @ world_mat @ camera_mat @ pixels_depth
    p_world = p_world[:, :3].permute(0, 2, 1)
    if is_numpy:
        p_world = p_world.numpy()
    return p_world
