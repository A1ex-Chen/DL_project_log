def eval_class(gt_annos, dt_annos, current_class, difficulty, metric,
    min_overlap, compute_aos=False, num_parts=50):
    """Kitti eval. Only support 2d/bev/3d/aos eval for now.
    Args:
        gt_annos: dict, must from get_label_annos() in kitti_common.py
        dt_annos: dict, must from get_label_annos() in kitti_common.py
        current_class: int, 0: car, 1: pedestrian, 2: cyclist
        difficulty: int. eval difficulty, 0: easy, 1: normal, 2: hard
        metric: eval type. 0: bbox, 1: bev, 2: 3d
        min_overlap: float, min overlap. official:
            [[0.7, 0.5, 0.5], [0.7, 0.5, 0.5], [0.7, 0.5, 0.5]]
            format: [metric, class]. choose one from matrix above.
        num_parts: int. a parameter for fast calculate algorithm

    Returns:
        dict of recall, precision and aos
    """
    assert len(gt_annos) == len(dt_annos)
    num_examples = len(gt_annos)
    split_parts = get_split_parts(num_examples, num_parts)
    thresholdss = []
    rets = calculate_iou_partly(dt_annos, gt_annos, metric, num_parts)
    overlaps, parted_overlaps, total_dt_num, total_gt_num = rets
    rets = _prepare_data(gt_annos, dt_annos, current_class, difficulty)
    (gt_datas_list, dt_datas_list, ignored_gts, ignored_dets, dontcares,
        total_dc_num, total_num_valid_gt) = rets
    for i in range(len(gt_annos)):
        rets = compute_statistics_jit(overlaps[i], gt_datas_list[i],
            dt_datas_list[i], ignored_gts[i], ignored_dets[i], dontcares[i],
            metric, min_overlap=min_overlap, thresh=0.0, compute_fp=False)
        tp, fp, fn, similarity, thresholds = rets
        thresholdss += thresholds.tolist()
    thresholdss = np.array(thresholdss)
    thresholds = get_thresholds(thresholdss, total_num_valid_gt)
    thresholds = np.array(thresholds)
    pr = np.zeros([len(thresholds), 4])
    idx = 0
    for j, num_part in enumerate(split_parts):
        gt_datas_part = np.concatenate(gt_datas_list[idx:idx + num_part], 0)
        dt_datas_part = np.concatenate(dt_datas_list[idx:idx + num_part], 0)
        dc_datas_part = np.concatenate(dontcares[idx:idx + num_part], 0)
        ignored_dets_part = np.concatenate(ignored_dets[idx:idx + num_part], 0)
        ignored_gts_part = np.concatenate(ignored_gts[idx:idx + num_part], 0)
        fused_compute_statistics(parted_overlaps[j], pr, total_gt_num[idx:
            idx + num_part], total_dt_num[idx:idx + num_part], total_dc_num
            [idx:idx + num_part], gt_datas_part, dt_datas_part,
            dc_datas_part, ignored_gts_part, ignored_dets_part, metric,
            min_overlap=min_overlap, thresholds=thresholds, compute_aos=
            compute_aos)
        idx += num_part
    N_SAMPLE_PTS = 41
    precision = np.zeros([N_SAMPLE_PTS])
    recall = np.zeros([N_SAMPLE_PTS])
    aos = np.zeros([N_SAMPLE_PTS])
    for i in range(len(thresholds)):
        recall[i] = pr[i, 0] / (pr[i, 0] + pr[i, 2])
        precision[i] = pr[i, 0] / (pr[i, 0] + pr[i, 1])
        if compute_aos:
            aos[i] = pr[i, 3] / (pr[i, 0] + pr[i, 1])
    for i in range(len(thresholds)):
        precision[i] = np.max(precision[i:])
        recall[i] = np.max(recall[i:])
        if compute_aos:
            aos[i] = np.max(aos[i:])
    ret_dict = {'recall': recall, 'precision': precision, 'orientation': aos}
    return ret_dict
