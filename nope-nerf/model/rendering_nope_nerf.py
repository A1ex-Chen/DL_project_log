def nope_nerf(self, pixels, depth, camera_mat, world_mat, scale_mat,
    add_noise=False, it=100000, eval_=False):
    batch_size, n_points, _ = pixels.shape
    device = self._device
    full_steps = self.cfg['num_points']
    dist_alpha = self.cfg['dist_alpha']
    sample_option = self.cfg['sample_option']
    use_dir = self.cfg['use_ray_dir']
    normalise_ray = self.cfg['normalise_ray']
    normal_loss = self.cfg['normal_loss']
    outside_steps = self.cfg['outside_steps']
    depth_range = torch.tensor(self.depth_range)
    n_max_network_queries = self.n_max_network_queries
    camera_world = origin_to_world(n_points, camera_mat, world_mat, scale_mat)
    points_world = transform_to_world(pixels, depth, camera_mat, world_mat,
        scale_mat)
    d_i_gt = torch.norm(points_world - camera_world, p=2, dim=-1)
    pixels_world = image_points_to_world(pixels, camera_mat, world_mat,
        scale_mat)
    ray_vector = pixels_world - camera_world
    ray_vector_norm = ray_vector.norm(2, 2)
    if normalise_ray:
        ray_vector = ray_vector / ray_vector.norm(2, 2).unsqueeze(-1)
    else:
        d_i_gt = d_i_gt / ray_vector_norm
    d_i = d_i_gt.clone()
    mask_zero_occupied = d_i == 0
    mask_pred = get_mask(d_i)
    dists = torch.ones_like(d_i).to(device)
    dists[mask_pred] = d_i[mask_pred]
    dists[mask_zero_occupied] = 0.0
    network_object_mask = mask_pred & ~mask_zero_occupied
    network_object_mask = network_object_mask[0]
    dists = dists[0]
    camera_world = camera_world.reshape(-1, 3)
    ray_vector = ray_vector.reshape(-1, 3)
    points = camera_world + ray_vector * dists.unsqueeze(-1)
    points = points.view(-1, 3)
    z_val = torch.linspace(0.0, 1.0, steps=full_steps - outside_steps,
        device=device)
    z_val = z_val.view(1, 1, -1).repeat(batch_size, n_points, 1)
    if sample_option == 'ndc':
        z_val, pts, ray_vector_fg = self.sample_ndc(camera_mat,
            camera_world, ray_vector, z_val, depth_range=[0.0, 1.0])
    elif sample_option == 'uniform':
        z_val, pts, ray_vector_fg = self.sample_uniform(camera_world,
            ray_vector, z_val, add_noise, depth_range)
    if not use_dir:
        ray_vector_fg = torch.ones_like(ray_vector_fg)
    noise = not eval_
    rgb_fg, logits_alpha_fg = [], []
    for i in range(0, pts.shape[0], n_max_network_queries):
        rgb_i, logits_alpha_i = self.model(pts[i:i + n_max_network_queries],
            ray_vector_fg[i:i + n_max_network_queries], return_addocc=True,
            noise=noise, it=it)
        rgb_fg.append(rgb_i)
        logits_alpha_fg.append(logits_alpha_i)
    rgb_fg = torch.cat(rgb_fg, dim=0)
    logits_alpha_fg = torch.cat(logits_alpha_fg, dim=0)
    rgb = rgb_fg.reshape(batch_size * n_points, full_steps, 3)
    alpha = logits_alpha_fg.view(batch_size * n_points, full_steps)
    if dist_alpha:
        t_vals = z_val.view(batch_size * n_points, full_steps)
        deltas = t_vals[:, 1:] - t_vals[:, :-1]
        dist_far = torch.empty(size=(batch_size * n_points, 1), dtype=torch
            .float32, device=dists.device).fill_(10000000000.0)
        deltas = torch.cat([deltas, dist_far], dim=-1)
        alpha = 1 - torch.exp(-1.0 * alpha * deltas)
        alpha[:, -1] = 1.0
    weights = alpha * torch.cumprod(torch.cat([torch.ones((rgb.shape[0], 1),
        device=device), 1.0 - alpha + epsilon], -1), -1)[:, :-1]
    rgb_values = torch.sum(weights.unsqueeze(-1) * rgb, dim=-2)
    dist_pred = torch.sum(weights.unsqueeze(-1) * z_val, dim=-2).squeeze(-1)
    if not eval_ and normal_loss:
        surface_mask = network_object_mask.view(-1)
        surface_points = points[surface_mask]
        N = surface_points.shape[0]
        surface_points_neig = surface_points + (torch.rand_like(
            surface_points) - 0.5) * 0.01
        pp = torch.cat([surface_points, surface_points_neig], dim=0)
        g = self.model.gradient(pp, it)
        normals_ = g[:, 0, :] / (g[:, 0, :].norm(2, dim=1).unsqueeze(-1) + 
            10 ** -5)
        diff_norm = torch.norm(normals_[:N] - normals_[N:], dim=-1)
    else:
        diff_norm = None
    if self.white_background:
        acc_map = torch.sum(weights, -1)
        rgb_values = rgb_values + (1.0 - acc_map.unsqueeze(-1))
    d_i_gt = d_i_gt[0]
    if eval_ and normalise_ray:
        dist_pred = dist_pred / ray_vector_norm[0]
        dists = dists / ray_vector_norm[0]
        d_i_gt = d_i_gt / ray_vector_norm[0]
    dist_rendered_masked = dist_pred[network_object_mask]
    dist_dpt_masked = d_i_gt[network_object_mask]
    if sample_option == 'ndc':
        dist_dpt_masked = 1 - 1 / dist_dpt_masked
    out_dict = {'rgb': rgb_values.reshape(batch_size, -1, 3), 'z_vals':
        z_val.squeeze(-1), 'normal': diff_norm, 'depth_pred':
        dist_rendered_masked, 'depth_gt': dist_dpt_masked, 'alpha': alpha}
    return out_dict
