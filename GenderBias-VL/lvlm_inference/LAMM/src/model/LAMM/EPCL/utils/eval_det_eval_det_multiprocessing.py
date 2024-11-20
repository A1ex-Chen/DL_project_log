def eval_det_multiprocessing(pred_all, gt_all, ovthresh=0.25, use_07_metric
    =False, get_iou_func=get_iou_obb):
    """Generic functions to compute precision/recall for object detection
    for multiple classes.
    Input:
        pred_all: map of {img_id: [(classname, bbox, score)]}
        gt_all: map of {img_id: [(classname, bbox)]}
        ovthresh: scalar, iou threshold
        use_07_metric: bool, if true use VOC07 11 point method
    Output:
        rec: {classname: rec}
        prec: {classname: prec_all}
        ap: {classname: scalar}
    """
    pred = {}
    gt = {}
    for img_id in pred_all.keys():
        for classname, bbox, score in pred_all[img_id]:
            if classname not in pred:
                pred[classname] = {}
            if img_id not in pred[classname]:
                pred[classname][img_id] = []
            if classname not in gt:
                gt[classname] = {}
            if img_id not in gt[classname]:
                gt[classname][img_id] = []
            pred[classname][img_id].append((bbox, score))
    for img_id in gt_all.keys():
        for classname, bbox in gt_all[img_id]:
            if classname not in gt:
                gt[classname] = {}
            if img_id not in gt[classname]:
                gt[classname][img_id] = []
            gt[classname][img_id].append(bbox)
    rec = {}
    prec = {}
    ap = {}
    p = Pool(processes=10)
    ret_values = p.map(eval_det_cls_wrapper, [(pred[classname], gt[
        classname], ovthresh, use_07_metric, get_iou_func) for classname in
        gt.keys() if classname in pred])
    p.close()
    for i, classname in enumerate(gt.keys()):
        if classname in pred:
            rec[classname], prec[classname], ap[classname] = ret_values[i]
        else:
            rec[classname] = 0
            prec[classname] = 0
            ap[classname] = 0
    return rec, prec, ap