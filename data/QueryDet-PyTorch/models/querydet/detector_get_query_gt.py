@torch.no_grad()
def get_query_gt(self, small_anchor_centers, targets):
    small_gt_cls = []
    for lind, anchor_center in enumerate(small_anchor_centers):
        per_layer_small_gt = []
        for target_per_image in targets:
            target_box_scales = get_box_scales(target_per_image.gt_boxes)
            small_inds = (target_box_scales < self.small_obj_scale[lind][1]
                ) & (target_box_scales >= self.small_obj_scale[lind][0])
            small_boxes = target_per_image[small_inds]
            center_dis, minarg = get_anchor_center_min_dis(small_boxes.
                gt_boxes.get_centers(), anchor_center)
            small_obj_target = torch.zeros_like(center_dis)
            if len(small_boxes) != 0:
                min_small_target_scale = target_box_scales[small_inds][minarg]
                small_obj_target[center_dis < min_small_target_scale * self
                    .small_center_dis_coeff[lind]] = 1
            per_layer_small_gt.append(small_obj_target)
        small_gt_cls.append(torch.stack(per_layer_small_gt))
    return small_gt_cls
