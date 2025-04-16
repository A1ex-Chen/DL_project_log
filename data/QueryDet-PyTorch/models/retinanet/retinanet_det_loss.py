def det_loss(self, gt_classes, gt_anchors_deltas, pred_logits, pred_deltas,
    alphas, gammas, cls_weights, reg_weights):

    def convert_gt_cls(logits, gt_class, f_idxs):
        gt_classes_target = torch.zeros_like(logits)
        gt_classes_target[f_idxs, gt_class[f_idxs]] = 1
        return gt_classes_target
    assert len(cls_weights) == len(pred_logits)
    assert len(cls_weights) == len(reg_weights)
    pred_logits, pred_deltas = permute_all_to_NHWA_K_not_concat(pred_logits,
        pred_deltas, self.num_classes)
    lengths = [x.shape[0] for x in pred_logits]
    start_inds = [0] + [sum(lengths[:i]) for i in range(1, len(lengths))]
    end_inds = [sum(lengths[:i + 1]) for i in range(len(lengths))]
    gt_classes = gt_classes.flatten()
    gt_anchors_deltas = gt_anchors_deltas.view(-1, 4)
    valid_idxs = gt_classes >= 0
    foreground_idxs = (gt_classes >= 0) & (gt_classes != self.num_classes)
    num_foreground = foreground_idxs.sum().item()
    get_event_storage().put_scalar('num_foreground', num_foreground)
    self.loss_normalizer = (self.loss_normalizer_momentum * self.
        loss_normalizer + (1 - self.loss_normalizer_momentum) * num_foreground)
    gt_clsses_list = [gt_classes[s:e] for s, e in zip(start_inds, end_inds)]
    gt_anchors_deltas_list = [gt_anchors_deltas[s:e] for s, e in zip(
        start_inds, end_inds)]
    valid_idxs_list = [valid_idxs[s:e] for s, e in zip(start_inds, end_inds)]
    foreground_idxs_list = [foreground_idxs[s:e] for s, e in zip(start_inds,
        end_inds)]
    loss_cls = [(w * sigmoid_focal_loss_jit(x[v], convert_gt_cls(x, g, f)[v
        ].detach(), alpha=alpha, gamma=gamma, reduction='sum')) for w, x, g,
        v, f, alpha, gamma in zip(cls_weights, pred_logits, gt_clsses_list,
        valid_idxs_list, foreground_idxs_list, alphas, gammas)]
    loss_box_reg = [(w * smooth_l1_loss(x[f], g[f].detach(), beta=self.
        smooth_l1_loss_beta, reduction='sum')) for w, x, g, f in zip(
        reg_weights, pred_deltas, gt_anchors_deltas_list, foreground_idxs_list)
        ]
    loss_cls = sum(loss_cls) / max(1.0, self.loss_normalizer)
    loss_box_reg = sum(loss_box_reg) / max(1.0, self.loss_normalizer)
    return {'loss_cls': loss_cls, 'loss_box_reg': loss_box_reg}
