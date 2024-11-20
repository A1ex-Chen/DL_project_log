def ap_score(bboxes_det_per_class: list[list[float, float, float, float,
    float, str, str]], bboxes_gt_per_class: list[list[float, float, float,
    float, float, str, str]], iou_threshold: float) ->dict[str, Any]:
    """Calculate average precision.

    Args:
        bboxes_det_per_class(list): bbox of detected object per class.
        bboxes_gt_per_class(list): bbox of ground truth object per class.
        IOUThreshold(float): iou threshold. it is usually set to 50%, 75% or 95%.

    Returns:
        ap(dict): dict containing information about average precision

    Note:
        bboxes_det_per_class: [bbox_det_1, bbox_det_2, ...]
        bboxes_gt_per_class: [bbox_gt_1, bbox_gt_2, ...]

        #The elements of each bbox variable are as follows, each element basically corresponding to a property of the BoundingBox class of Object-Detection-Metrics.
        https://github.com/rafaelpadilla/Object-Detection-Metrics/blob/master/lib/BoundingBox.py

        ----
        bbox_det_n(tuple): (xmin, ymin, width, height, confidence, class_id, image_name)
        bbox_gt_n(tuple): (xmin, ymin, width, height, 1.0, class_id, image_name)

        xmin(float): xmin
        ymin(float): ymin
        width(float): width
        height(float): height
        confidence(float): class confidence
        class_id(str): class id
        image_name(str): image name

        #index variable, this is written as a global variable in the `def main()` function.
        X_INDEX = 0
        Y_INDEX = 1
        W_INDEX = 2
        H_INDEX = 3
        CONFIDENCE_INDEX = 4
        CLASS_ID_INDEX = 5
        IMAGE_NAME_INDEX = 6
    """
    assert len(bboxes_gt_per_class
        ) != 0, 'It must contain at least one Grand Truth.'
    class_id = bboxes_gt_per_class[0][CLASS_ID_INDEX]
    n_dets = len(bboxes_det_per_class)
    n_gts = len(bboxes_gt_per_class)
    if len(bboxes_det_per_class) == 0:
        return {'class': class_id, 'precision': [], 'recall': [], 'AP': 0.0,
            'interpolated precision': [], 'interpolated recall': [],
            'total positives': 0, 'total TP': 0, 'total FP': 0}
    validate_bboxes(bboxes_det_per_class, is_gt=False)
    validate_bboxes(bboxes_gt_per_class, is_gt=True)
    for bbox_det in bboxes_det_per_class:
        assert bbox_det[CLASS_ID_INDEX
            ] == class_id, f'class_id must be the same for all bboxes, but {bbox_det[CLASS_ID_INDEX]} found.'
    for bbox_gt in bboxes_gt_per_class:
        assert bbox_gt[CLASS_ID_INDEX
            ] == class_id, f'class_id must be the same for all bboxes, but {bbox_gt[CLASS_ID_INDEX]} found.'
    gts: dict[str, Any] = {}
    for bbox_gt in bboxes_gt_per_class:
        image_name = bbox_gt[IMAGE_NAME_INDEX]
        gts[image_name] = gts.get(image_name, []) + [bbox_gt]
    bboxes_det_per_class = sorted(bboxes_det_per_class, key=lambda x: x[
        CONFIDENCE_INDEX], reverse=True)
    det = {key: np.zeros(len(gt)) for key, gt in gts.items()}
    iouMax_list = []
    TP = np.zeros(len(bboxes_det_per_class))
    FP = np.zeros(len(bboxes_det_per_class))
    for d, bbox_det in enumerate(bboxes_det_per_class):
        image_name = bbox_det[IMAGE_NAME_INDEX]
        gt_bboxes = gts.get(image_name, [])
        iouMax = sys.float_info.min
        bbox_det = [bbox_det[X_INDEX], bbox_det[Y_INDEX], bbox_det[W_INDEX],
            bbox_det[H_INDEX]]
        bbox_det = convert_to_x1y1x2y2(bbox_det)
        for j, bbox_gt in enumerate(gt_bboxes):
            bbox_gt = [bbox_gt[X_INDEX], bbox_gt[Y_INDEX], bbox_gt[W_INDEX],
                bbox_gt[H_INDEX]]
            bbox_gt = convert_to_x1y1x2y2(bbox_gt)
            iou = iou_score(bbox_det, bbox_gt)
            if iou > iouMax:
                iouMax = iou
                jmax = j
                iouMax_list.append(iouMax)
        if iouMax >= iou_threshold and det[image_name][jmax] != 1:
            TP[d] = 1
            det[image_name][jmax] = 1
        else:
            FP[d] = 1
    acc_FP = np.cumsum(FP)
    acc_TP = np.cumsum(TP)
    rec = acc_TP / n_gts
    prec = np.divide(acc_TP, acc_FP + acc_TP)
    [ap_, mpre_, mrec_, _] = ElevenPointInterpolatedAP(rec, prec)
    return {'class': class_id, 'precision': list(prec), 'recall': list(rec),
        'AP': ap_, 'interpolated precision': mpre_, 'interpolated recall':
        mrec_, 'total positives': n_dets, 'total TP': int(np.sum(TP)),
        'total FP': int(np.sum(FP))}
