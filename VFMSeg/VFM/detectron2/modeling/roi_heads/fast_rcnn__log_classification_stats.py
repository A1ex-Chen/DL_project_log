def _log_classification_stats(pred_logits, gt_classes, prefix='fast_rcnn'):
    """
    Log the classification metrics to EventStorage.

    Args:
        pred_logits: Rx(K+1) logits. The last column is for background class.
        gt_classes: R labels
    """
    num_instances = gt_classes.numel()
    if num_instances == 0:
        return
    pred_classes = pred_logits.argmax(dim=1)
    bg_class_ind = pred_logits.shape[1] - 1
    fg_inds = (gt_classes >= 0) & (gt_classes < bg_class_ind)
    num_fg = fg_inds.nonzero().numel()
    fg_gt_classes = gt_classes[fg_inds]
    fg_pred_classes = pred_classes[fg_inds]
    num_false_negative = (fg_pred_classes == bg_class_ind).nonzero().numel()
    num_accurate = (pred_classes == gt_classes).nonzero().numel()
    fg_num_accurate = (fg_pred_classes == fg_gt_classes).nonzero().numel()
    storage = get_event_storage()
    storage.put_scalar(f'{prefix}/cls_accuracy', num_accurate / num_instances)
    if num_fg > 0:
        storage.put_scalar(f'{prefix}/fg_cls_accuracy', fg_num_accurate /
            num_fg)
        storage.put_scalar(f'{prefix}/false_negative', num_false_negative /
            num_fg)
