def eval_images(self, data, render_dir, fxfy, lpips_vgg_fn, logger,
    min_depth=0.1, max_depth=20, it=0):
    self.renderer.eval()
    img_gt, depth_gt, camera_mat, scale_mat, img_idx = self.process_data_dict(
        data)
    img_idx = int(img_idx)
    img_gt = img_gt.squeeze(0).permute(1, 2, 0)
    depth_gt = depth_gt.squeeze(0).numpy()
    mask = (depth_gt > min_depth) * (depth_gt < max_depth)
    if self.use_learnt_poses:
        c2w = self.c2ws[img_idx]
        world_mat = torch.inverse(c2w).unsqueeze(0)
    if self.use_learnt_focal:
        camera_mat = torch.tensor([[[fxfy[0], 0, 0, 0], [0, -fxfy[1], 0, 0],
            [0, 0, -1, 0], [0, 0, 0, 1]]]).to(self.device)
    h, w = self.resolution
    p_loc, pixels = arange_pixels(resolution=(h, w))
    pixels = pixels.to(self.device)
    depth_input = torch.zeros(1, 1, h, w).to(self.device)
    depth_input = get_tensor_values(depth_input, pixels.clone(), mode=
        'nearest', scale=True, detach=False)
    depth_input = torch.ones_like(depth_input)
    with torch.no_grad():
        rgb_pred = []
        depth_pred = []
        for i, (pixels_i, depth_i) in enumerate(zip(torch.split(pixels,
            self.points_batch_size, dim=1), torch.split(depth_input, self.
            points_batch_size, dim=1))):
            out_dict = self.renderer(pixels_i, depth_i, camera_mat,
                world_mat, scale_mat, self.render_type, eval_=True, it=it,
                add_noise=False)
            rgb_pred_i = out_dict['rgb']
            rgb_pred.append(rgb_pred_i)
            depth_pred_i = out_dict['depth_pred']
            depth_pred.append(depth_pred_i)
        rgb_pred = torch.cat(rgb_pred, dim=1)
        img_out = rgb_pred.view(h, w, 3)
        depth_pred = torch.cat(depth_pred, dim=0)
        depth_pred = depth_pred.view(h, w).detach().cpu().numpy()
        depth_out = depth_pred
    mse = F.mse_loss(img_out, img_gt).item()
    psnr = mse2psnr(mse)
    ssim = pytorch_ssim.ssim(img_out.permute(2, 0, 1).unsqueeze(0), img_gt.
        permute(2, 0, 1).unsqueeze(0)).item()
    lpips_loss = lpips_vgg_fn(img_out.permute(2, 0, 1).unsqueeze(0).
        contiguous(), img_gt.permute(2, 0, 1).unsqueeze(0).contiguous(),
        normalize=True).item()
    tqdm.write('{0:4d} img: PSNR: {1:.2f}, SSIM: {2:.2f},  LPIPS {3:.2f}'.
        format(img_idx, psnr, ssim, lpips_loss))
    gt_height, gt_width = depth_gt.shape[:2]
    depth_out = cv2.resize(depth_out, (gt_width, gt_height), interpolation=
        cv2.INTER_NEAREST)
    img_out_dir = os.path.join(render_dir, 'img_out')
    depth_out_dir = os.path.join(render_dir, 'depth_out')
    img_gt_dir = os.path.join(render_dir, 'img_gt_out')
    if not os.path.exists(img_out_dir):
        os.makedirs(img_out_dir)
    if not os.path.exists(depth_out_dir):
        os.makedirs(depth_out_dir)
    if not os.path.exists(img_gt_dir):
        os.makedirs(img_gt_dir)
    depth_out = np.clip(255.0 / depth_out.max() * (depth_out - depth_out.
        min()), 0, 255).astype(np.uint8)
    img_out = (img_out.cpu().numpy() * 255).astype(np.uint8)
    img_gt = (img_gt.cpu().numpy() * 255).astype(np.uint8)
    imageio.imwrite(os.path.join(img_out_dir, str(img_idx).zfill(4) +
        '.png'), img_out)
    imageio.imwrite(os.path.join(depth_out_dir, str(img_idx).zfill(4) +
        '.png'), depth_out)
    imageio.imwrite(os.path.join(img_gt_dir, str(img_idx).zfill(4) + '.png'
        ), img_gt)
    depth_out = depth_out[mask]
    depth_gt = depth_gt[mask]
    img_dict = {'img': img_out, 'depth': depth_out, 'mse': mse, 'psnr':
        psnr, 'ssim': ssim, 'lpips': lpips_loss, 'depth_pred': depth_out,
        'depth_gt': depth_gt}
    return img_dict
