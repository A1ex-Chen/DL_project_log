def filter_gt_boxes(gt_boxes, gt_labels, used_classes):
    mask = np.array([(l in used_classes) for l in gt_labels], dtype=np.bool)
    return mask
