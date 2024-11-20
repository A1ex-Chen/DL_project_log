def origin_to_world(n_points, camera_mat, world_mat, scale_mat, invert=True):
    """ Transforms origin (camera location) to world coordinates.

    Args:
        n_points (int): how often the transformed origin is repeated in the
            form (batch_size, n_points, 3)
        camera_mat (tensor): camera matrix
        world_mat (tensor): world matrix
        scale_mat (tensor): scale matrix
        invert (bool): whether to invert the matrices (default: true)
    """
    batch_size = camera_mat.shape[0]
    device = camera_mat.device
    p = torch.zeros(batch_size, 4, n_points, device=device)
    p[:, -1] = 1.0
    if invert:
        camera_mat = torch.inverse(camera_mat)
        world_mat = torch.inverse(world_mat)
        scale_mat = torch.inverse(scale_mat)
    p_world = scale_mat @ world_mat @ camera_mat @ p
    p_world = p_world[:, :3].permute(0, 2, 1)
    return p_world
