def ap_score_range(bboxes_det_per_class: list[float, float, float, float,
    float, str, str], bboxes_gt_per_class: list[float, float, float, float,
    float, str, str], start_threshold: float=0.5, end_threshold: float=0.95,
    step: float=0.05) ->float:
    """Calculate average precision in the specified range.

    Args:
        bboxes_det_per_class(list): bbox of detected object per class.
        bboxes_gt_per_class(list): bbox of ground truth object per class.
        start_threshold(float): start threshold of IOU. default is 0.5.
        end_threshold(float): end threshold of IOU. default is 0.95.
        step(float): step of updating threshold. default is 0.05.

    Returns:
        ap_results(list): list of average precision in the specified range.
        ap_range(float): average of ap in the specified range.

    """
    ap_list = []
    for iou_threshold in np.arange(start_threshold, end_threshold + step, step
        ):
        ap_result = ap_score(bboxes_det_per_class, bboxes_gt_per_class,
            iou_threshold)
        ap_list.append(ap_result['AP'])
    ap_range = np.mean(ap_list)
    return ap_range
