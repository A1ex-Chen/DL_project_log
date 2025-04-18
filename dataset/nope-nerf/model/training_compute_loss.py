def compute_loss(self, data, eval_mode=False, it=None, epoch=None,
    scheduling_start=None, out_render_path=None):
    """ Compute the loss.

        Args:
            data (dict): data dictionary
            eval_mode (bool): whether to use eval mode
            it (int): training iteration
            epoch(int): current number of epochs
            scheduling_start(int): num of epochs to start scheduling
            out_render_path(str): path to save rendered images
        """
    weights = {}
    weights_name_list = ['rgb_weight', 'depth_weight', 'pc_weight',
        'rgb_s_weight', 'depth_consistency_weight', 'weight_dist_2nd_loss',
        'weight_dist_1st_loss']
    weights_list = [self.anneal(getattr(self, w)[0], getattr(self, w)[1],
        scheduling_start, self.annealing_epochs, epoch) for w in
        weights_name_list]
    rgb_loss_type = ('l1' if epoch < self.annealing_epochs +
        scheduling_start else 'l2')
    for i, weight in enumerate(weights_list):
        weight_name = weights_name_list[i]
        weights[weight_name] = weight
    render_model = weights['rgb_weight'] != 0.0 or weights['depth_weight'
        ] != 0.0
    use_ref_imgs = weights['pc_weight'] != 0.0 or weights['rgb_s_weight'
        ] != 0.0
    n_points = self.n_training_points
    nl = self.nearest_limit
    img, depth_input, camera_mat_gt, scale_mat, img_idx = (self.
        process_data_dict(data))
    if use_ref_imgs:
        ref_img, depth_ref, ref_idx = self.process_data_reference(data)
    device = self.device
    batch_size, _, h, w = img.shape
    batch_size, _, h_depth, w_depth = depth_input.shape
    kwargs = dict()
    kwargs['t_list'] = self.pose_param_net.get_t()
    kwargs['weights'] = weights
    kwargs['rgb_loss_type'] = rgb_loss_type
    if self.pose_param_net is not None:
        num_cams = self.pose_param_net.num_cams
        c2w = self.pose_param_net(img_idx)
        world_mat = torch.inverse(c2w).unsqueeze(0)
    if self.distortion_net is not None:
        scale_input, shift_input = self.distortion_net(img_idx)
        if self.shift_first:
            depth_input = (depth_input + shift_input) * scale_input
        else:
            depth_input = depth_input * scale_input + shift_input
    if self.optimizer_focal:
        fxfy = self.focal_net(0)
        pad = torch.zeros(4).to(device)
        one = torch.tensor([1]).to(device)
        camera_mat = torch.cat([fxfy[0:1], pad, -fxfy[1:2], pad, -one, pad,
            one])
        camera_mat = camera_mat.view(1, 4, 4)
    else:
        camera_mat = camera_mat_gt
    ray_idx = torch.randperm(h * w, device=device)[:n_points]
    img_flat = img.view(batch_size, 3, h * w).permute(0, 2, 1)
    rgb_gt = img_flat[:, ray_idx]
    p_full = arange_pixels((h, w), batch_size, device=device)[1]
    p = p_full[:, ray_idx]
    pix = ray_idx
    if render_model:
        out_dict = self.model(p, pix, camera_mat, world_mat, scale_mat,
            self.rendering_technique, it=it, eval_mode=eval_mode, depth_img
            =depth_input, img_size=(h, w))
        rendered_rgb = out_dict['rgb']
        rendered_depth = out_dict['depth_pred']
        gt_depth = out_dict['depth_gt']
    else:
        rendered_rgb = None
        rendered_depth = None
        gt_depth = None
    if use_ref_imgs:
        c2w_ref = self.pose_param_net(ref_idx)
        if self.distortion_net is not None:
            scale_ref, shift_ref = self.distortion_net(ref_idx)
            if self.shift_first:
                depth_ref = scale_ref * (depth_ref + shift_ref)
            else:
                depth_ref = scale_ref * depth_ref + shift_ref
        if self.detach_ref_img:
            c2w_ref = c2w_ref.detach()
            scale_ref = scale_ref.detach()
            shift_ref = shift_ref.detach()
            depth_ref = depth_ref.detach()
        ref_Rt = torch.inverse(c2w_ref).unsqueeze(0)
        if img_idx < num_cams - 1:
            d1 = depth_input
            d2 = depth_ref
            img1 = img
            img2 = ref_img
            Rt_rel_12 = ref_Rt @ torch.inverse(world_mat)
            R_rel_12 = Rt_rel_12[:, :3, :3]
            t_rel_12 = Rt_rel_12[:, :3, 3]
            scale2 = scale_ref
        else:
            d1 = depth_ref
            d2 = depth_input
            img1 = ref_img
            img2 = img
            Rt_rel_12 = world_mat @ torch.inverse(ref_Rt)
            R_rel_12 = Rt_rel_12[:, :3, :3]
            t_rel_12 = Rt_rel_12[:, :3, 3]
            scale2 = scale_input
        ratio = self.pc_ratio
        sample_resolution = int(h_depth / ratio), int(w_depth / ratio)
        pixel_locations, p_pc = arange_pixels(resolution=sample_resolution,
            device=device)
        d1 = F.interpolate(d1, sample_resolution, mode='nearest')
        d2 = F.interpolate(d2, sample_resolution, mode='nearest')
        d1[d1 < nl] = nl
        d2[d2 < nl] = nl
        pc1 = transform_to_world(p_pc, d1.view(1, -1, 1), camera_mat)
        pc2 = transform_to_world(p_pc, d2.view(1, -1, 1), camera_mat)
        if weights['rgb_s_weight'] != 0.0:
            img1 = F.interpolate(img1, sample_resolution, mode='bilinear')
            img2 = F.interpolate(img2, sample_resolution, mode='bilinear')
            rgb_pc1 = get_tensor_values(img1, p_pc, mode='bilinear', scale=
                False, detach=False, detach_p=False, align_corners=True)
            if self.detach_rgbs_scale:
                pc1_ = pc1.detach().clone()
                pc1_rotated = pc1_ @ R_rel_12.transpose(1, 2) + t_rel_12
            else:
                pc1_rotated = pc1 @ R_rel_12.transpose(1, 2) + t_rel_12
            mask_pc1_invalid = (-pc1_rotated[:, :, 2:] < nl).expand_as(
                pc1_rotated)
            pc1_rotated[mask_pc1_invalid] = nl
            p_reprojected, valid_mask = project_to_cam(pc1_rotated,
                camera_mat, device)
            rgb_pc1_proj = get_tensor_values(img2, p_reprojected, mode=
                'bilinear', scale=False, detach=False, detach_p=False,
                align_corners=True)
            rgb_pc1 = rgb_pc1.view(batch_size, sample_resolution[0],
                sample_resolution[1], 3)
            rgb_pc1_proj = rgb_pc1_proj.view(batch_size, sample_resolution[
                0], sample_resolution[1], 3)
            valid_mask = valid_mask.view(batch_size, sample_resolution[0],
                sample_resolution[1], 1)
            kwargs['rgb_pc1'] = rgb_pc1
            kwargs['rgb_pc1_proj'] = rgb_pc1_proj
            kwargs['valid_points'] = valid_mask
            if it % self.vis_reprojection_every == 0:
                Image.fromarray((rgb_pc1[0] * 255).detach().cpu().numpy().
                    astype(np.uint8)).convert('RGB').save(os.path.join(
                    out_render_path, '%d_%04d_img1.png' % (it, img_idx)))
                Image.fromarray((rgb_pc1_proj[0] * 255).detach().cpu().
                    numpy().astype(np.uint8)).convert('RGB').save(os.path.
                    join(out_render_path, '%d_%04d_img2.png' % (it, img_idx)))
        pc1 = pc1 @ R_rel_12.transpose(1, 2) + t_rel_12
        if self.scale_pcs:
            pc1 = pc1 / scale2
            pc2 = pc2 / scale2
        kwargs['X'] = pc1
        kwargs['Y'] = pc2
        kwargs['sample_resolution'] = sample_resolution
        kwargs['p_2d'] = pixel_locations
    if render_model and self.detach_gt_depth:
        gt_depth = gt_depth.detach()
    loss_dict = self.loss(rendered_rgb, rgb_gt, rendered_depth, gt_depth,
        **kwargs)
    if self.optimizer_focal:
        loss_dict['focalx'] = fxfy[0] / camera_mat_gt[0, 0, 0]
        loss_dict['focaly'] = fxfy[1] / camera_mat_gt[0, 1, 1]
    loss_dict['scale'] = scale_input
    loss_dict['shift'] = shift_input
    return loss_dict
