def do_eval(gt_annos, dt_annos, current_class, min_overlaps, compute_aos=False
    ):
    mAP_bbox = []
    mAP_aos = []
    for i in range(3):
        ret = eval_class(gt_annos, dt_annos, current_class, i, 0,
            min_overlaps[0], compute_aos)
        mAP_bbox.append(get_mAP(ret['precision']))
        if compute_aos:
            mAP_aos.append(get_mAP(ret['orientation']))
    mAP_bev = []
    for i in range(3):
        ret = eval_class(gt_annos, dt_annos, current_class, i, 1,
            min_overlaps[1])
        mAP_bev.append(get_mAP(ret['precision']))
    mAP_3d = []
    for i in range(3):
        ret = eval_class(gt_annos, dt_annos, current_class, i, 2,
            min_overlaps[2])
        mAP_3d.append(get_mAP(ret['precision']))
    return mAP_bbox, mAP_bev, mAP_3d, mAP_aos
