def __call__(self, outputs, targets, epoch_num, step_num, batch_height,
    batch_width):
    feats, pred_scores, pred_distri = outputs
    if all(feat.shape[2:] == cfsize for feat, cfsize in zip(feats, self.
        cached_feat_sizes)):
        anchors, anchor_points, n_anchors_list, stride_tensor = (self.
            cached_anchors)
    else:
        self.cached_feat_sizes = [feat.shape[2:] for feat in feats]
        anchors, anchor_points, n_anchors_list, stride_tensor = (
            generate_anchors(feats, self.fpn_strides, self.grid_cell_size,
            self.grid_cell_offset, device=feats[0].device))
        self.cached_anchors = (anchors, anchor_points, n_anchors_list,
            stride_tensor)
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
    try:
        if epoch_num < self.warmup_epoch:
            target_labels, target_bboxes, target_scores, fg_mask = (self.
                warmup_assigner(anchors, n_anchors_list, gt_labels,
                gt_bboxes, mask_gt, pred_bboxes.detach() * stride_tensor))
        else:
            target_labels, target_bboxes, target_scores, fg_mask = (self.
                formal_assigner(pred_scores.detach(), pred_bboxes.detach() *
                stride_tensor, anchor_points, gt_labels, gt_bboxes, mask_gt))
    except RuntimeError:
        print(
            'OOM RuntimeError is raised due to the huge memory cost during label assignment.                     CPU mode is applied in this batch. If you want to avoid this issue,                     try to reduce the batch size or image size.'
            )
        torch.cuda.empty_cache()
        print('------------CPU Mode for This Batch-------------')
        if epoch_num < self.warmup_epoch:
            _anchors = anchors.cpu().float()
            _n_anchors_list = n_anchors_list
            _gt_labels = gt_labels.cpu().float()
            _gt_bboxes = gt_bboxes.cpu().float()
            _mask_gt = mask_gt.cpu().float()
            _pred_bboxes = pred_bboxes.detach().cpu().float()
            _stride_tensor = stride_tensor.cpu().float()
            target_labels, target_bboxes, target_scores, fg_mask = (self.
                warmup_assigner(_anchors, _n_anchors_list, _gt_labels,
                _gt_bboxes, _mask_gt, _pred_bboxes * _stride_tensor))
        else:
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
    if target_scores_sum > 1:
        loss_cls /= target_scores_sum
    loss_iou, loss_dfl = self.bbox_loss(pred_distri, pred_bboxes,
        anchor_points_s, target_bboxes, target_scores, target_scores_sum,
        fg_mask)
    loss = self.loss_weight['class'] * loss_cls + self.loss_weight['iou'
        ] * loss_iou + self.loss_weight['dfl'] * loss_dfl
    return loss, torch.cat(((self.loss_weight['iou'] * loss_iou).unsqueeze(
        0), (self.loss_weight['dfl'] * loss_dfl).unsqueeze(0), (self.
        loss_weight['class'] * loss_cls).unsqueeze(0))).detach()
