def phong_renderer(self, pixels, camera_mat, world_mat, scale_mat, it):
    batch_size, num_pixels, _ = pixels.shape
    device = self._device
    rad = self.cfg['radius']
    n_points = num_pixels
    pixels_world = image_points_to_world(pixels, camera_mat, world_mat,
        scale_mat)
    camera_world = origin_to_world(num_pixels, camera_mat, world_mat, scale_mat
        )
    ray_vector = pixels_world - camera_world
    ray_vector = ray_vector / ray_vector.norm(2, 2).unsqueeze(-1)
    light_source = camera_world[0, 0]
    light = (light_source / light_source.norm(2)).unsqueeze(1).cuda()
    diffuse_per = torch.Tensor([0.7, 0.7, 0.7]).float()
    ambiant = torch.Tensor([0.3, 0.3, 0.3]).float()
    self.model.eval()
    with torch.no_grad():
        d_i = self.ray_marching(camera_world, ray_vector, self.model,
            n_secant_steps=8, n_steps=[int(512), int(512) + 1], rad=rad)
    d_i = d_i.detach()
    mask_zero_occupied = d_i == 0
    mask_pred = get_mask(d_i).detach()
    with torch.no_grad():
        dists = torch.ones_like(d_i).to(device)
        dists[mask_pred] = d_i[mask_pred].detach()
        dists[mask_zero_occupied] = 0.0
        network_object_mask = mask_pred & ~mask_zero_occupied
        network_object_mask = network_object_mask[0]
        dists = dists[0]
        camera_world = camera_world.reshape(-1, 3)
        ray_vector = ray_vector.reshape(-1, 3)
        points = camera_world + ray_vector * dists.unsqueeze(-1)
        points = points.view(-1, 3)
        view_vol = -1 * ray_vector.view(-1, 3)
        rgb_values = torch.ones_like(points).float().cuda()
        surface_points = points[network_object_mask]
        surface_view_vol = view_vol[network_object_mask]
        grad = []
        for pnts in torch.split(surface_points, 1000000, dim=0):
            grad.append(self.model.gradient(pnts, it)[:, 0, :].detach())
            torch.cuda.empty_cache()
        grad = torch.cat(grad, 0)
        surface_normals = grad / grad.norm(2, 1, keepdim=True)
    diffuse = torch.mm(surface_normals, light).clamp_min(0).repeat(1, 3
        ) * diffuse_per.unsqueeze(0).cuda()
    rgb_values[network_object_mask] = (ambiant.unsqueeze(0).cuda() + diffuse
        ).clamp_max(1.0)
    with torch.no_grad():
        rgb_val = torch.zeros(batch_size * n_points, 3, device=device)
        rgb_val[network_object_mask] = self.model(surface_points,
            surface_view_vol)
    out_dict = {'rgb': rgb_values.reshape(batch_size, -1, 3), 'normal':
        None, 'rgb_surf': rgb_val.reshape(batch_size, -1, 3)}
    return out_dict
