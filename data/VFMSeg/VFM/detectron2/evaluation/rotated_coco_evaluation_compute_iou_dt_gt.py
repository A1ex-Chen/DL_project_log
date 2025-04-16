def compute_iou_dt_gt(self, dt, gt, is_crowd):
    if self.is_rotated(dt) or self.is_rotated(gt):
        assert all(c == 0 for c in is_crowd)
        dt = RotatedBoxes(self.boxlist_to_tensor(dt, output_box_dim=5))
        gt = RotatedBoxes(self.boxlist_to_tensor(gt, output_box_dim=5))
        return pairwise_iou_rotated(dt, gt)
    else:
        return maskUtils.iou(dt, gt, is_crowd)
