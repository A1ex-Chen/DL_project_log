def transform_to_camera_space(p_world, camera_mat, world_mat, scale_mat):
    """ Transforms world points to camera space.
        Args:
        p_world (tensor): world points tensor of size B x N x 3
        camera_mat (tensor): camera matrix
        world_mat (tensor): world matrix
        scale_mat (tensor): scale matrix
    """
    batch_size, n_p, _ = p_world.shape
    device = p_world.device
    p_world = torch.cat([p_world, torch.ones(batch_size, n_p, 1).to(device)
        ], dim=-1).permute(0, 2, 1)
    p_cam = camera_mat @ world_mat @ scale_mat @ p_world
    p_cam = p_cam[:, :3].permute(0, 2, 1)
    return p_cam
