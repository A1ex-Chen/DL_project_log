def get_DPT_reprojection_loss(self, rgb, rgb_refs, valid_points,
    rgb_img_refs_ori):
    cfg = self.cfg
    loss = 0
    for rgb_ref, rgb_img_ref_ori in zip(rgb_refs, rgb_img_refs_ori):
        diff_img = (rgb - rgb_ref).abs()
        diff_img = diff_img.clamp(0, 1)
        if cfg['with_auto_mask'] == True:
            auto_mask = (diff_img.mean(dim=1, keepdim=True) < (rgb -
                rgb_img_ref_ori).abs().mean(dim=1, keepdim=True)).float()
            auto_mask = auto_mask * valid_points
            valid_points = auto_mask
        if cfg['with_ssim'] == True:
            ssim_map = compute_ssim_loss(rgb, rgb_ref)
            diff_img = 0.15 * diff_img + 0.85 * ssim_map
        loss = loss + self.mean_on_mask(diff_img, valid_points)
    loss = loss / len(rgb_refs)
    return loss
