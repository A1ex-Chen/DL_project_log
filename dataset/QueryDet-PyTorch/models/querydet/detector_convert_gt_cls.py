def convert_gt_cls(logits, gt_class, f_idxs):
    gt_classes_target = torch.zeros_like(logits)
    gt_classes_target[f_idxs, gt_class[f_idxs]] = 1
    return gt_classes_target
