def do_eval_v2(gt_annos, dt_annos, current_classes, min_overlaps,
    compute_aos=False, difficultys=[0, 1, 2]):
    ret = eval_class_v3(gt_annos, dt_annos, current_classes, difficultys, 0,
        min_overlaps, compute_aos)
    mAP_bbox = get_mAP_v2(ret['precision'])
    mAP_aos = None
    if compute_aos:
        mAP_aos = get_mAP_v2(ret['orientation'])
    ret = eval_class_v3(gt_annos, dt_annos, current_classes, difficultys, 1,
        min_overlaps)
    mAP_bev = get_mAP_v2(ret['precision'])
    ret = eval_class_v3(gt_annos, dt_annos, current_classes, difficultys, 2,
        min_overlaps)
    mAP_3d = get_mAP_v2(ret['precision'])
    return mAP_bbox, mAP_bev, mAP_3d, mAP_aos
