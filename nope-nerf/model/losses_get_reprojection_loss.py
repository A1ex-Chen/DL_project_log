def get_reprojection_loss(self, rgb, rgb_refs, valid_points, rgb_refs_ori):
    cfg = self.cfg
    loss = 0
    for rgb_ref, rgb_ref_ori in zip(rgb_refs, rgb_refs_ori):
        diff_img = (rgb - rgb_ref).abs()
        if cfg['with_auto_mask'] == True:
            auto_mask = (diff_img.mean(dim=-1, keepdim=True) < (rgb -
                rgb_ref_ori).abs().mean(dim=-1, keepdim=True)).float(
                ) * valid_points
            valid_points = auto_mask
        loss = loss + self.mean_on_mask(diff_img, valid_points)
    loss = loss / len(rgb_refs)
    return loss
