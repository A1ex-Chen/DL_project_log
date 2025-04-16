def get_losses(self, bbox_preds, cls_preds, obj_preds, origin_preds,
    org_xy_shifts, xy_shifts, expanded_strides, center_ltrbes, whwh, labels,
    dtype):
    nlabel = labels[:, 0].long().bincount(minlength=cls_preds.shape[0]).tolist(
        )
    batch_gt_classes = labels[:, 1].type_as(cls_preds).contiguous()
    batch_org_gt_bboxes = labels[:, 2:6].contiguous()
    batch_org_gt_bboxes.mul_(whwh)
    batch_gt_bboxes = torch.empty_like(batch_org_gt_bboxes)
    batch_gt_half_wh = batch_org_gt_bboxes[:, 2:] / 2
    batch_gt_bboxes[:, :2] = batch_org_gt_bboxes[:, :2] - batch_gt_half_wh
    batch_gt_bboxes[:, 2:] = batch_org_gt_bboxes[:, :2] + batch_gt_half_wh
    batch_org_gt_bboxes = batch_org_gt_bboxes.type_as(bbox_preds)
    batch_gt_bboxes = batch_gt_bboxes.type_as(bbox_preds)
    del batch_gt_half_wh
    total_num_anchors = bbox_preds.shape[1]
    cls_targets = []
    reg_targets = []
    l1_targets = []
    fg_mask_inds = []
    num_fg = 0.0
    num_gts = 0
    index_offset = 0
    batch_size = bbox_preds.shape[0]
    for batch_idx in range(batch_size):
        num_gt = int(nlabel[batch_idx])
        if num_gt == 0:
            cls_target = bbox_preds.new_zeros((0, self.num_classes))
            reg_target = bbox_preds.new_zeros((0, 4))
            l1_target = bbox_preds.new_zeros((0, 4))
        else:
            _num_gts = num_gts + num_gt
            org_gt_bboxes_per_image = batch_org_gt_bboxes[num_gts:_num_gts]
            gt_bboxes_per_image = batch_gt_bboxes[num_gts:_num_gts]
            gt_classes = batch_gt_classes[num_gts:_num_gts]
            num_gts = _num_gts
            bboxes_preds_per_image = bbox_preds[batch_idx]
            cls_preds_per_image = cls_preds[batch_idx]
            obj_preds_per_image = obj_preds[batch_idx]
            try:
                (gt_matched_classes, fg_mask_ind, pred_ious_this_matching,
                    matched_gt_inds, num_fg_img) = (self.get_assignments(
                    num_gt, total_num_anchors, org_gt_bboxes_per_image,
                    gt_bboxes_per_image, gt_classes, self.num_classes,
                    bboxes_preds_per_image, cls_preds_per_image,
                    obj_preds_per_image, center_ltrbes, xy_shifts))
            except RuntimeError:
                LOGGER.warning(
                    'OOM RuntimeError is raised due to the huge memory cost during label assignment.                            CPU mode is applied in this batch. If you want to avoid this issue,                            try to reduce the batch size or image size.'
                    )
                torch.cuda.empty_cache()
                LOGGER.warning(
                    '------------CPU Mode for This Batch-------------')
                _org_gt_bboxes_per_image = org_gt_bboxes_per_image.cpu().float(
                    )
                _gt_bboxes_per_image = gt_bboxes_per_image.cpu().float()
                _bboxes_preds_per_image = bboxes_preds_per_image.cpu().float()
                _cls_preds_per_image = cls_preds_per_image.cpu().float()
                _obj_preds_per_image = obj_preds_per_image.cpu().float()
                _gt_classes = gt_classes.cpu().float()
                _center_ltrbes = center_ltrbes.cpu().float()
                _xy_shifts = xy_shifts.cpu()
                (gt_matched_classes, fg_mask_ind, pred_ious_this_matching,
                    matched_gt_inds, num_fg_img) = (self.get_assignments(
                    num_gt, total_num_anchors, _org_gt_bboxes_per_image,
                    _gt_bboxes_per_image, _gt_classes, self.num_classes,
                    _bboxes_preds_per_image, _cls_preds_per_image,
                    _obj_preds_per_image, _center_ltrbes, _xy_shifts))
                gt_matched_classes = gt_matched_classes.cuda()
                fg_mask_ind = fg_mask_ind.cuda()
                pred_ious_this_matching = pred_ious_this_matching.cuda()
                matched_gt_inds = matched_gt_inds.cuda()
            torch.cuda.empty_cache()
            num_fg += num_fg_img
            cls_target = F.one_hot(gt_matched_classes.to(torch.int64), self
                .num_classes) * pred_ious_this_matching.view(-1, 1)
            reg_target = gt_bboxes_per_image[matched_gt_inds]
            if self.use_l1:
                l1_target = self.get_l1_target(bbox_preds.new_empty((
                    num_fg_img, 4)), org_gt_bboxes_per_image[
                    matched_gt_inds], expanded_strides[0][fg_mask_ind],
                    xy_shifts=org_xy_shifts[0][fg_mask_ind])
            if index_offset > 0:
                fg_mask_ind.add_(index_offset)
            fg_mask_inds.append(fg_mask_ind)
        index_offset += total_num_anchors
        cls_targets.append(cls_target)
        reg_targets.append(reg_target)
        if self.use_l1:
            l1_targets.append(l1_target)
    cls_targets = torch.cat(cls_targets, 0)
    reg_targets = torch.cat(reg_targets, 0)
    fg_mask_inds = torch.cat(fg_mask_inds, 0)
    if self.use_l1:
        l1_targets = torch.cat(l1_targets, 0)
    num_fg = max(num_fg, 1)
    loss_iou = self.iou_loss(bbox_preds.view(-1, 4)[fg_mask_inds],
        reg_targets, True).sum() / num_fg
    obj_preds = obj_preds.view(-1, 1)
    obj_targets = torch.zeros_like(obj_preds).index_fill_(0, fg_mask_inds, 1)
    loss_obj = self.bcewithlog_loss(obj_preds, obj_targets).sum() / num_fg
    loss_cls = self.bcewithlog_loss(cls_preds.view(-1, self.num_classes)[
        fg_mask_inds], cls_targets).sum() / num_fg
    if self.use_l1:
        loss_l1 = self.l1_loss(origin_preds.view(-1, 4)[fg_mask_inds],
            l1_targets).sum() / num_fg
    else:
        loss_l1 = torch.zeros_like(loss_iou)
    reg_weight = 5.0
    loss_iou = reg_weight * loss_iou
    loss = loss_iou + loss_obj + loss_cls + loss_l1
    return loss, loss_iou, loss_obj, loss_cls, loss_l1, num_fg / max(num_gts, 1
        )
