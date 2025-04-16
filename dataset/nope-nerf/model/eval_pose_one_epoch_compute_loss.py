def compute_loss(self, data, eval_mode=False, it=100000):
    """ Compute the loss.

        Args:
            data (dict): data dictionary
            eval_mode (bool): whether to use eval mode
            it (int): training iteration
        """
    n_points = self.n_points
    img, depth_img, camera_mat, scale_mat, img_idx = self.process_data_dict(
        data)
    device = self.device
    batch_size, _, h, w = img.shape
    c2w = self.pose_param_net(img_idx)
    world_mat = torch.inverse(c2w).unsqueeze(0)
    if self.focal_net is not None:
        fxfy = self.focal_net(0)
        pad = torch.zeros(4)
        one = torch.tensor([1])
        camera_mat = torch.cat([fxfy[0:1], pad, -fxfy[1:2], pad, -one, pad,
            one]).to(device)
        camera_mat = camera_mat.view(1, 4, 4)
    ray_idx = torch.randperm(h * w, device=device)[:n_points]
    img_flat = img.view(batch_size, 3, h * w).permute(0, 2, 1)
    rgb_gt = img_flat[:, ray_idx]
    p_full = arange_pixels((h, w), batch_size)[1].to(device)
    p = p_full[:, ray_idx]
    pix = ray_idx
    out_dict = self.model(p, pix, camera_mat, world_mat, scale_mat, self.
        rendering_technique, it=it, eval_mode=True, depth_img=depth_img,
        add_noise=False, img_size=(h, w))
    loss_dict = self.loss(out_dict['rgb'], rgb_gt)
    return loss_dict
