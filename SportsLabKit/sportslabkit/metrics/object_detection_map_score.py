def map_score(bboxes_det: (pd.DataFrame | BBoxDataFrame | list | tuple),
    bboxes_gt: (pd.DataFrame | BBoxDataFrame | list | tuple), iou_threshold:
    float) ->float:
    """Calculate mean average precision.

    Args:
        det_df(pd.DataFrame): dataframe of detected object.
        gt_df(pd.DataFrame): dataframe of ground truth object.
        IOUThreshold(float): iou threshold

    Returns:
        map(float): mean average precision
    """
    bboxes_det = convert_bboxes(bboxes_det)
    bboxes_gt = convert_bboxes(bboxes_gt)
    ap_list = []
    class_list = []
    for bbox_gt in bboxes_gt:
        if bbox_gt[CLASS_ID_INDEX] not in class_list:
            class_list.append(bbox_gt[CLASS_ID_INDEX])
    classes = sorted(class_list)
    for class_id in classes:
        bboxes_det_per_class = [detection_per_class for detection_per_class in
            bboxes_det if detection_per_class[CLASS_ID_INDEX] == class_id]
        bboxes_gt_per_class = [groundTruth_per_class for
            groundTruth_per_class in bboxes_gt if groundTruth_per_class[
            CLASS_ID_INDEX] == class_id]
        ap = ap_score(bboxes_det_per_class, bboxes_gt_per_class, iou_threshold)
        ap_list.append(ap['AP'])
    map = np.mean(ap_list)
    return map
