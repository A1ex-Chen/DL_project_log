def sample_uniform(self, camera_world, ray_vector, z_val, add_noise,
    depth_range):
    batch_size, n_points, full_steps = z_val.shape
    z_val = depth_range[0] * (1.0 - z_val) + depth_range[1] * z_val
    if add_noise:
        di_mid = 0.5 * (z_val[:, :, 1:] + z_val[:, :, :-1])
        di_high = torch.cat([di_mid, z_val[:, :, -1:]], dim=-1)
        di_low = torch.cat([z_val[:, :, :1], di_mid], dim=-1)
        noise = torch.rand(batch_size, n_points, full_steps, device=self.
            _device)
        z_val = di_low + (di_high - di_low) * noise
    pts = camera_world.unsqueeze(-2) + ray_vector.unsqueeze(-2
        ) * z_val.unsqueeze(-1)
    pts = pts.reshape(-1, 3)
    ray_vector_fg = ray_vector.unsqueeze(-2).repeat(1, 1, full_steps, 1)
    ray_vector_fg = -1 * ray_vector_fg.reshape(-1, 3)
    z_val = z_val.view(-1, full_steps, 1)
    return z_val, pts, ray_vector_fg
