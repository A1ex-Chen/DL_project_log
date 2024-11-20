def filter_kitti_anno(image_anno, used_classes, used_difficulty=None,
    dontcare_iou=None):
    if not isinstance(used_classes, (list, tuple, np.ndarray)):
        used_classes = [used_classes]
    img_filtered_annotations = {}
    relevant_annotation_indices = [i for i, x in enumerate(image_anno[
        'name']) if x in used_classes]
    for key in image_anno.keys():
        img_filtered_annotations[key] = image_anno[key][
            relevant_annotation_indices]
    if used_difficulty is not None:
        relevant_annotation_indices = [i for i, x in enumerate(
            img_filtered_annotations['difficulty']) if x in used_difficulty]
        for key in image_anno.keys():
            img_filtered_annotations[key] = img_filtered_annotations[key][
                relevant_annotation_indices]
    if 'DontCare' in used_classes and dontcare_iou is not None:
        dont_care_indices = [i for i, x in enumerate(
            img_filtered_annotations['name']) if x == 'DontCare']
        all_boxes = img_filtered_annotations['bbox']
        ious = iou(all_boxes, all_boxes[dont_care_indices])
        if ious.size > 0:
            boxes_to_remove = np.amax(ious, axis=1) > dontcare_iou
            for key in image_anno.keys():
                img_filtered_annotations[key] = img_filtered_annotations[key][
                    np.logical_not(boxes_to_remove)]
    return img_filtered_annotations
