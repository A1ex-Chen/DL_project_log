def get_rgb_s_loss(self, rgb1, rgb2, valid_points):
    diff_img = (rgb1 - rgb2).abs()
    diff_img = diff_img.clamp(0, 1)
    if self.cfg['with_ssim'] == True:
        ssim_map = compute_ssim_loss(rgb1, rgb2)
        diff_img = 0.15 * diff_img + 0.85 * ssim_map
    loss = self.mean_on_mask(diff_img, valid_points)
    return loss
