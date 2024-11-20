def process_labels_for_training(image_info, boxes, classes, score_targets,
    box_targets, max_num_instances, min_level, max_level):
    labels = {}
    boxes = preprocess_ops.pad_to_fixed_size(boxes, -1, [max_num_instances, 4])
    classes = preprocess_ops.pad_to_fixed_size(classes, -1, [
        max_num_instances, 1])
    for level in range(min_level, max_level + 1):
        labels['score_targets_%d' % level] = score_targets[level]
        labels['box_targets_%d' % level] = box_targets[level]
    labels['gt_boxes'] = boxes
    labels['gt_classes'] = classes
    return labels
