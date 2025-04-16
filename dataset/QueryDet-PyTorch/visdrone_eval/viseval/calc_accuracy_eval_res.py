def eval_res(gt0, dt0, thr):
    """
    :param gt0: np.array[ng, 5], ground truth results [x, y, w, h, ignore]
    :param dt0: np.array[nd, 5], detection results [x, y, w, h, score]
    :param thr: float, IoU threshold
    :return gt1: np.array[ng, 5], gt match types
             dt1: np.array[nd, 6], dt match types
    """
    nd = len(dt0)
    ng = len(gt0)
    dt = dt0[dt0[:, 4].argsort()[::-1]]
    gt_ignore_mask = gt0[:, 4] == 1
    gt = gt0[np.logical_not(gt_ignore_mask)]
    ig = gt0[gt_ignore_mask]
    ig[:, 4] = -ig[:, 4]
    dt_format = dt[:, :4].copy()
    gt_format = gt[:, :4].copy()
    ig_format = ig[:, :4].copy()
    dt_format[:, 2:] += dt_format[:, :2]
    gt_format[:, 2:] += gt_format[:, :2]
    ig_format[:, 2:] += ig_format[:, :2]
    iou_dtgt = bbox_overlaps(dt_format, gt_format, mode='iou')
    iof_dtig = bbox_overlaps(dt_format, gt_format, mode='iof')
    oa = np.concatenate((iou_dtgt, iof_dtig), axis=1)
    dt1 = np.concatenate((dt, np.zeros((nd, 1), dtype=dt.dtype)), axis=1)
    gt1 = np.concatenate((gt, ig), axis=0)
    for d in range(nd):
        bst_oa = thr
        bstg = -1
        bstm = 0
        for g in range(ng):
            m = gt1[g, 4]
            if m == 1:
                continue
            if bstm != 0 and m == -1:
                break
            if oa[d, g] < bst_oa:
                continue
            bst_oa = oa[d, g]
            bstg = g
            bstm = 1 if m == 0 else -1
        dt1[d, 5] = bstm
        if bstm == 1:
            gt1[bstg, 4] = 1
    return gt1, dt1
