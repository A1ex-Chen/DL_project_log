def map_score_range(bboxes_det: (pd.DataFrame | BBoxDataFrame | list |
    tuple), bboxes_gt: (pd.DataFrame | BBoxDataFrame | list | tuple),
    start_threshold: float=0.5, end_threshold: float=0.95, step: float=0.05
    ) ->float:
    """Calculate mean average precision.

    Args:
        det_df(pd.DataFrame): dataframe of detected object.
        gt_df(pd.DataFrame): dataframe of ground truth object.
        start_threshold(float): start threshold of IOU. default is 0.5.
        end_threshold(float): end threshold of IOU. default is 0.95.
        step(float): step of updating threshold. default is 0.05.

    Returns:
        map_range(float): average of map in the specified range. (0.5 to 0.95 in increments of 0.05)

    """
    map_list = []
    for iou_threshold in np.arange(start_threshold, end_threshold + step, step
        ):
        map_result = map_score(bboxes_det, bboxes_gt, iou_threshold)
        map_list.append(map_result)
    map_range = np.mean(map_list)
    return map_range
