def calc_accuracy(num_imgs, all_gt, all_det, per_class=False):
    """
    :param num_imgs: int
    :param all_gt: list of np.array[m, 8], [:, 4] == 1 indicates ignored regions,
                    which should be dropped before calling this function
    :param all_det: list of np.array[m, 6], truncation and occlusion not necessary
    :param per_class:
    """
    assert num_imgs == len(all_gt) == len(all_det)
    ap = np.zeros((10, 10), dtype=np.float32)
    ar = np.zeros((10, 10, 4), dtype=np.float32)
    eval_class = []
    print('')
    for id_class in range(1, 11):
        print('evaluating object category {}/10...'.format(id_class))
        for gt in all_gt:
            if np.any(gt[:, 5] == id_class):
                eval_class.append(id_class - 1)
        x = 0
        for thr in np.linspace(0.5, 0.95, num=10):
            y = 0
            for max_dets in (1, 10, 100, 500):
                gt_match = []
                det_match = []
                for gt, det in zip(all_gt, all_det):
                    det_limited = det[:min(len(det), max_dets)]
                    mask_gt_cur_class = gt[:, 5] == id_class
                    mask_det_cur_class = det_limited[:, 5] == id_class
                    gt0 = gt[mask_gt_cur_class, :5]
                    dt0 = det_limited[mask_det_cur_class, :5]
                    gt1, dt1 = eval_res(gt0, dt0, thr)
                    gt_match.append(gt1[:, 4])
                    det_match.append(dt1[:, 4:6])
                gt_match = np.concatenate(gt_match, axis=0)
                det_match = np.concatenate(det_match, axis=0)
                idrank = det_match[:, 0].argsort()[::-1]
                tp = np.cumsum(det_match[idrank, 1] == 1)
                rec = tp / max(1, len(gt_match))
                if len(rec):
                    ar[id_class - 1, x, y] = np.max(rec) * 100
                y += 1
            fp = np.cumsum(det_match[idrank, 1] == 0)
            prec = tp / (fp + tp).clip(min=1)
            ap[id_class - 1, x] = voc_ap(rec, prec) * 100
            x += 1
    ap_all = np.mean(ap[eval_class, :])
    ap_50 = np.mean(ap[eval_class, 0])
    ap_75 = np.mean(ap[eval_class, 5])
    ar_1 = np.mean(ar[eval_class, :, 0])
    ar_10 = np.mean(ar[eval_class, :, 1])
    ar_100 = np.mean(ar[eval_class, :, 2])
    ar_500 = np.mean(ar[eval_class, :, 3])
    results = ap_all, ap_50, ap_75, ar_1, ar_10, ar_100, ar_500
    if per_class:
        ap_classwise = np.mean(ap, axis=1)
        results += ap_classwise,
    print(
        'Evaluation completed. The performance of the detector is presented as follows.'
        )
    return results
