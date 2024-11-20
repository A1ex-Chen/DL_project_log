@torch.no_grad()
def get_assignments(self, num_gt, total_num_anchors,
    org_gt_bboxes_per_image, gt_bboxes_per_image, gt_classes, num_classes,
    bboxes_preds_per_image, cls_preds_per_image, obj_preds_per_image,
    center_ltrbes, xy_shifts):
    fg_mask_inds, is_in_boxes_and_center = self.get_in_boxes_info(
        org_gt_bboxes_per_image, gt_bboxes_per_image, center_ltrbes,
        xy_shifts, total_num_anchors, num_gt)
    bboxes_preds_per_image = bboxes_preds_per_image[fg_mask_inds]
    cls_preds_ = cls_preds_per_image[fg_mask_inds]
    obj_preds_ = obj_preds_per_image[fg_mask_inds]
    num_in_boxes_anchor = bboxes_preds_per_image.shape[0]
    pair_wise_ious = self.bboxes_iou(gt_bboxes_per_image,
        bboxes_preds_per_image, True, inplace=True)
    pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-08)
    cls_preds_ = cls_preds_.float().sigmoid_().unsqueeze(0).expand(num_gt,
        num_in_boxes_anchor, num_classes)
    obj_preds_ = obj_preds_.float().sigmoid_().unsqueeze(0).expand(num_gt,
        num_in_boxes_anchor, 1)
    cls_preds_ = (cls_preds_ * obj_preds_).sqrt_()
    del obj_preds_
    gt_cls_per_image = F.one_hot(gt_classes.to(torch.int64), num_classes
        ).float()
    gt_cls_per_image = gt_cls_per_image[:, None, :].expand(num_gt,
        num_in_boxes_anchor, num_classes)
    with autocast(enabled=False):
        pair_wise_cls_loss = F.binary_cross_entropy(cls_preds_,
            gt_cls_per_image, reduction='none').sum(-1)
    del cls_preds_, gt_cls_per_image
    cost = (pair_wise_cls_loss + 3.0 * pair_wise_ious_loss + 100000.0 * ~
        is_in_boxes_and_center)
    del pair_wise_cls_loss, pair_wise_ious_loss, is_in_boxes_and_center
    (num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds,
        fg_mask_inds) = (self.dynamic_k_matching(cost, pair_wise_ious,
        gt_classes, num_gt, fg_mask_inds))
    del cost, pair_wise_ious
    return (gt_matched_classes, fg_mask_inds, pred_ious_this_matching,
        matched_gt_inds, num_fg)
