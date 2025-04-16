@staticmethod
def get_in_boxes_info(org_gt_bboxes_per_image, gt_bboxes_per_image,
    center_ltrbes, xy_shifts, total_num_anchors, num_gt):
    xy_centers_per_image = xy_shifts.expand(num_gt, total_num_anchors, 2)
    gt_bboxes_per_image = gt_bboxes_per_image[:, None, :].expand(num_gt,
        total_num_anchors, 4)
    b_lt = xy_centers_per_image - gt_bboxes_per_image[..., :2]
    b_rb = gt_bboxes_per_image[..., 2:] - xy_centers_per_image
    bbox_deltas = torch.cat([b_lt, b_rb], 2)
    is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0
    is_in_boxes_all = is_in_boxes.sum(dim=0) > 0
    center_ltrbes = center_ltrbes.expand(num_gt, total_num_anchors, 4)
    org_gt_xy_center = org_gt_bboxes_per_image[:, 0:2]
    org_gt_xy_center = torch.cat([-org_gt_xy_center, org_gt_xy_center], dim=-1)
    org_gt_xy_center = org_gt_xy_center[:, None, :].expand(num_gt,
        total_num_anchors, 4)
    center_deltas = org_gt_xy_center + center_ltrbes
    is_in_centers = center_deltas.min(dim=-1).values > 0.0
    is_in_centers_all = is_in_centers.sum(dim=0) > 0
    is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all
    is_in_boxes_and_center = is_in_boxes[:, is_in_boxes_anchor
        ] & is_in_centers[:, is_in_boxes_anchor]
    return torch.nonzero(is_in_boxes_anchor)[..., 0], is_in_boxes_and_center
