def ap_per_class_box_and_mask(tp_m, tp_b, conf, pred_cls, target_cls, plot=
    False, save_dir='.', names=()):
    """
    Args:
        tp_b: tp of boxes.
        tp_m: tp of masks.
        other arguments see `func: ap_per_class`.
    """
    results_boxes = ap_per_class(tp_b, conf, pred_cls, target_cls, plot=
        plot, save_dir=save_dir, names=names, prefix='Box')[2:]
    results_masks = ap_per_class(tp_m, conf, pred_cls, target_cls, plot=
        plot, save_dir=save_dir, names=names, prefix='Mask')[2:]
    results = {'boxes': {'p': results_boxes[0], 'r': results_boxes[1], 'ap':
        results_boxes[3], 'f1': results_boxes[2], 'ap_class': results_boxes
        [4]}, 'masks': {'p': results_masks[0], 'r': results_masks[1], 'ap':
        results_masks[3], 'f1': results_masks[2], 'ap_class': results_masks[4]}
        }
    return results
