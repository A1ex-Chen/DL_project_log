def __call__(self, outputs, t_outputs, s_featmaps, t_featmaps, targets,
    epoch_num, max_epoch, temperature, step_num, batch_height, batch_width):
    feats, pred_scores, pred_distri, pred_lrtb = outputs
    t_feats, t_pred_scores, t_pred_distri = t_outputs[0], t_outputs[-2
        ], t_outputs[-1]
    anchors, anchor_points, n_anchors_list, stride_tensor = generate_anchors(
        feats, self.fpn_strides, self.grid_cell_size, self.grid_cell_offset,
        device=feats[0].device)
    t_anchors, t_anchor_points, t_n_anchors_list, t_stride_tensor = (
        generate_anchors(t_feats, self.fpn_strides, self.grid_cell_size,
        self.grid_cell_offset, device=feats[0].device))
    assert pred_scores.type() == pred_distri.type()
    gt_bboxes_scale = torch.tensor([batch_width, batch_height, batch_width,
        batch_height]).type_as(pred_scores)
    batch_size = pred_scores.shape[0]
    targets = self.preprocess(targets, batch_size, gt_bboxes_scale)
    gt_labels = targets[:, :, :1]
    gt_bboxes = targets[:, :, 1:]
    mask_gt = (gt_bboxes.sum(-1, keepdim=True) > 0).float()
    anchor_points_s = anchor_points / stride_tensor
    pred_bboxes = self.bbox_decode(anchor_points_s, pred_distri)
    pred_bboxes_lrtb = dist2bbox(pred_lrtb, anchor_points_s)
    t_anchor_points_s = t_anchor_points / t_stride_tensor
    t_pred_bboxes = self.bbox_decode(t_anchor_points_s, t_pred_distri)
    try:
        target_labels, target_bboxes, target_scores, fg_mask = (self.
            formal_assigner(pred_scores.detach(), pred_bboxes.detach() *
            stride_tensor, anchor_points, gt_labels, gt_bboxes, mask_gt))
    except RuntimeError:
        print(
            'OOM RuntimeError is raised due to the huge memory cost during label assignment.                     CPU mode is applied in this batch. If you want to avoid this issue,                     try to reduce the batch size or image size.'
            )
        torch.cuda.empty_cache()
        print('------------CPU Mode for This Batch-------------')
        _pred_scores = pred_scores.detach().cpu().float()
        _pred_bboxes = pred_bboxes.detach().cpu().float()
        _anchor_points = anchor_points.cpu().float()
        _gt_labels = gt_labels.cpu().float()
        _gt_bboxes = gt_bboxes.cpu().float()
        _mask_gt = mask_gt.cpu().float()
        _stride_tensor = stride_tensor.cpu().float()
        target_labels, target_bboxes, target_scores, fg_mask = (self.
            formal_assigner(_pred_scores, _pred_bboxes * _stride_tensor,
            _anchor_points, _gt_labels, _gt_bboxes, _mask_gt))
        target_labels = target_labels.cuda()
        target_bboxes = target_bboxes.cuda()
        target_scores = target_scores.cuda()
        fg_mask = fg_mask.cuda()
    if step_num % 10 == 0:
        torch.cuda.empty_cache()
    target_bboxes /= stride_tensor
    target_labels = torch.where(fg_mask > 0, target_labels, torch.full_like
        (target_labels, self.num_classes))
    one_hot_label = F.one_hot(target_labels.long(), self.num_classes + 1)[
        ..., :-1]
    loss_cls = self.varifocal_loss(pred_scores, target_scores, one_hot_label)
    target_scores_sum = target_scores.sum()
    if target_scores_sum > 0:
        loss_cls /= target_scores_sum
    loss_iou, loss_dfl, d_loss_dfl = self.bbox_loss(pred_distri,
        pred_bboxes_lrtb, pred_bboxes, t_pred_distri, t_pred_bboxes,
        temperature, anchor_points_s, target_bboxes, target_scores,
        target_scores_sum, fg_mask)
    logits_student = pred_scores
    logits_teacher = t_pred_scores
    distill_num_classes = self.num_classes
    d_loss_cls = self.distill_loss_cls(logits_student, logits_teacher,
        distill_num_classes, temperature)
    if self.distill_feat:
        d_loss_cw = self.distill_loss_cw(s_featmaps, t_featmaps)
    else:
        d_loss_cw = torch.tensor(0.0).to(feats[0].device)
    import math
    distill_weightdecay = (1 - math.cos(epoch_num * math.pi / max_epoch)
        ) / 2 * (0.01 - 1) + 1
    d_loss_dfl *= distill_weightdecay
    d_loss_cls *= distill_weightdecay
    d_loss_cw *= distill_weightdecay
    loss_cls_all = loss_cls + d_loss_cls * self.distill_weight['class']
    loss_dfl_all = loss_dfl + d_loss_dfl * self.distill_weight['dfl']
    loss = self.loss_weight['class'] * loss_cls_all + self.loss_weight['iou'
        ] * loss_iou + self.loss_weight['dfl'
        ] * loss_dfl_all + self.loss_weight['cwd'] * d_loss_cw
    return loss, torch.cat(((self.loss_weight['iou'] * loss_iou).unsqueeze(
        0), (self.loss_weight['dfl'] * loss_dfl_all).unsqueeze(0), (self.
        loss_weight['class'] * loss_cls_all).unsqueeze(0), (self.
        loss_weight['cwd'] * d_loss_cw).unsqueeze(0))).detach()
