def sample_ndc(self, camera_mat, camera_world, ray_vector, z_val,
    depth_range=[0.0, 1.0]):
    batch_size, n_points, full_steps = z_val.shape
    focal = torch.cat([camera_mat[:, 0, 0], camera_mat[:, 1, 1]])
    ray_ori_world, ray_dir_world = get_ndc_rays_fxfy(focal, 1.0, rays_o=
        camera_world, rays_d=ray_vector)
    z_val = depth_range[0] * (1.0 - z_val) + depth_range[1] * z_val
    pts = ray_ori_world.unsqueeze(-2) + ray_dir_world.unsqueeze(-2
        ) * z_val.unsqueeze(-1)
    pts = pts.reshape(-1, 3)
    ray_vector_fg = ray_vector.unsqueeze(-2).repeat(1, 1, full_steps, 1)
    ray_vector_fg = -1 * ray_vector_fg.reshape(-1, 3)
    z_val = z_val.view(-1, full_steps, 1)
    return z_val, pts, ray_vector_fg
