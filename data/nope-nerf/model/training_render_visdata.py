def render_visdata(self, data, resolution, it, out_render_path):
    img, dpt, camera_mat, scale_mat, img_idx = self.process_data_dict(data)
    h, w = resolution
    if self.pose_param_net:
        c2w = self.pose_param_net(img_idx)
        world_mat = torch.inverse(c2w).unsqueeze(0)
    if self.optimizer_focal:
        fxfy = self.focal_net(0)
        camera_mat = torch.tensor([[[fxfy[0], 0, 0, 0], [0, -fxfy[1], 0, 0],
            [0, 0, -1, 0], [0, 0, 0, 1]]]).to(self.device)
    p_idx = torch.arange(h * w).to(self.device)
    p_loc, pixels = arange_pixels(resolution=(h, w))
    pixels = pixels.to(self.device)
    depth_input = dpt
    with torch.no_grad():
        rgb_pred = []
        depth_pred = []
        for i, (pixels_i, p_idx_i) in enumerate(zip(torch.split(pixels, 
            1024, dim=1), torch.split(p_idx, 1024, dim=0))):
            out_dict = self.model(pixels_i, p_idx_i, camera_mat, world_mat,
                scale_mat, self.rendering_technique, add_noise=False,
                eval_mode=True, it=it, depth_img=depth_input, img_size=(h, w))
            rgb_pred_i = out_dict['rgb']
            rgb_pred.append(rgb_pred_i)
            depth_pred_i = out_dict['depth_pred']
            depth_pred.append(depth_pred_i)
        rgb_pred = torch.cat(rgb_pred, dim=1)
        depth_pred = torch.cat(depth_pred, dim=0)
        rgb_pred = rgb_pred.view(h, w, 3).detach().cpu().numpy()
        img_out = (rgb_pred * 255).astype(np.uint8)
        depth_pred_out = depth_pred.view(h, w).detach().cpu().numpy()
        imageio.imwrite(os.path.join(out_render_path, '%04d_depth.png' %
            img_idx), np.clip(255.0 / depth_pred_out.max() * (
            depth_pred_out - depth_pred_out.min()), 0, 255).astype(np.uint8))
        img1 = Image.fromarray(img_out.astype(np.uint8)).convert('RGB').save(os
            .path.join(out_render_path, '%04d_img.png' % img_idx))
    if self.vis_geo:
        with torch.no_grad():
            rgb_pred = [self.model(pixels_i, None, camera_mat, world_mat,
                scale_mat, 'phong_renderer', add_noise=False, eval_mode=
                True, it=it, depth_img=depth_input, img_size=(h, w))['rgb'] for
                i, pixels_i in enumerate(torch.split(pixels, 1024, dim=1))]
            rgb_pred = torch.cat(rgb_pred, dim=1).cpu()
            rgb_pred = rgb_pred.view(h, w, 3).detach().cpu().numpy()
            img_out = (rgb_pred * 255).astype(np.uint8)
            img1 = Image.fromarray(img_out.astype(np.uint8)).convert('RGB'
                ).save(os.path.join(out_render_path, '%04d_geo.png' % img_idx))
    return img_out.astype(np.uint8)
