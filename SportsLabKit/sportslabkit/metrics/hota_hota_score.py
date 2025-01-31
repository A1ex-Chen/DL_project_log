def hota_score(bboxes_track: BBoxDataFrame, bboxes_gt: BBoxDataFrame) ->dict[
    str, Any]:
    """Calculates the HOTA metrics for one sequence.

    Args:
        bboxes_track (BBoxDataFrame): Bbox Dataframe for tracking in 1 sequence
        bboxes_gt (BBoxDataFrame): Bbox Dataframe for ground truth in 1 sequence

    Returns:
        dict[str, Any]: HOTA metrics

    Note:
    The description of each evaluation indicator will be as follows:
    "HOTA" : The overall HOTA score.
    "DetA" : The detection accuracy.
    "AssA" : The association accuracy.
    "DetRe" : The detection recall.
    "DetPr" : The detection precision.
    "AssRe" : The association recall.
    "AssPr" : The association precision.
    "LocA" : The localization accuracy.
    "RHOTA" : The robust HOTA score.
    "HOTA(0)" : The overall HOTA score with a threshold of 0.5.
    "LocA(0)" : The localization accuracy with a threshold of 0.5.
    "HOTALocA(0)" : The overall HOTA score with a threshold of 0.5 and the localization accuracy with a threshold of 0.5.

    This is also based on the following original paper and the github repository.
    paper : https://link.springer.com/article/10.1007/s11263-020-01375-2
    code  : https://github.com/JonathonLuiten/TrackEval
    """
    data = to_mot_eval_format(bboxes_gt, bboxes_track)
    array_labels = np.arange(0.05, 0.99, 0.05)
    integer_array_fields = ['HOTA_TP', 'HOTA_FN', 'HOTA_FP']
    float_array_fields = ['HOTA', 'DetA', 'AssA', 'DetRe', 'DetPr', 'AssRe',
        'AssPr', 'LocA', 'RHOTA']
    float_fields = ['HOTA(0)', 'LocA(0)', 'HOTALocA(0)']
    res = {}
    for field in (float_array_fields + integer_array_fields):
        res[field] = np.zeros(len(array_labels), dtype=float)
    for field in float_fields:
        res[field] = 0
    if data['num_tracker_dets'] == 0:
        res['HOTA_FN'] = data['num_gt_dets'] * np.ones(len(array_labels),
            dtype=float)
        res['LocA'] = np.ones(len(array_labels), dtype=float)
        res['LocA(0)'] = 1.0
        hota_final_scores(res)
        return res
    if data['num_gt_dets'] == 0:
        res['HOTA_FP'] = data['num_tracker_dets'] * np.ones(len(
            array_labels), dtype=float)
        res['LocA'] = np.ones(len(array_labels), dtype=float)
        res['LocA(0)'] = 1.0
        hota_final_scores(res)
        return res
    potential_matches_count = np.zeros((data['num_gt_ids'], data[
        'num_tracker_ids']))
    gt_id_count = np.zeros((data['num_gt_ids'], 1))
    tracker_id_count = np.zeros((1, data['num_tracker_ids']))
    for t, (gt_ids_t, tracker_ids_t, gt_det_t, tracker_det_t) in enumerate(zip
        (data['gt_ids'], data['tracker_ids'], data['gt_dets'], data[
        'tracker_dets'])):
        similarity = data['similarity_scores'][t]
        sim_iou_denom = similarity.sum(0)[np.newaxis, :] + similarity.sum(1)[
            :, np.newaxis] - similarity
        if similarity.size:
            sim_iou = np.zeros_like(similarity)
            sim_iou_mask = sim_iou_denom > 0 + np.finfo('float').eps
            sim_iou[sim_iou_mask] = similarity[sim_iou_mask] / sim_iou_denom[
                sim_iou_mask]
            potential_matches_count[list(gt_ids_t[:, np.newaxis]), list(
                tracker_ids_t[np.newaxis, :])] += sim_iou
        count = np.array([[(0 if row[0] == -1 else 1) for _, row in
            enumerate(gt_det_t)]]).T
        gt_id_count[list(gt_ids_t)] += list(count)
        tracker_id_count[0, list(tracker_ids_t)] += [(0 if row[0] == -1 else
            1) for _, row in enumerate(tracker_det_t)]
    global_alignment_score = potential_matches_count / (gt_id_count +
        tracker_id_count - potential_matches_count)
    matches_counts = [np.zeros_like(potential_matches_count) for _ in
        array_labels]
    for t, (gt_ids_t, tracker_ids_t) in enumerate(zip(data['gt_ids'], data[
        'tracker_ids'])):
        if len(gt_ids_t) == 0:
            for a, alpha in enumerate(array_labels):
                res['HOTA_FP'][a] += len(tracker_ids_t)
            continue
        if len(tracker_ids_t) == 0:
            for a, alpha in enumerate(array_labels):
                res['HOTA_FN'][a] += len(gt_ids_t)
            continue
        similarity = data['similarity_scores'][t]
        score_mat = global_alignment_score[gt_ids_t[:, np.newaxis],
            tracker_ids_t[np.newaxis, :]] * similarity
        match_rows, match_cols = linear_sum_assignment(-score_mat)
        for a, alpha in enumerate(array_labels):
            actually_matched_mask = similarity[match_rows, match_cols
                ] >= alpha - np.finfo('float').eps
            alpha_match_rows = match_rows[actually_matched_mask]
            alpha_match_cols = match_cols[actually_matched_mask]
            num_matches = len(alpha_match_rows)
            res['HOTA_TP'][a] += num_matches
            res['HOTA_FN'][a] += len(gt_ids_t) - num_matches
            res['HOTA_FP'][a] += len(tracker_ids_t) - num_matches
            if num_matches > 0:
                res['LocA'][a] += sum(similarity[alpha_match_rows,
                    alpha_match_cols])
                matches_counts[a][gt_ids_t[alpha_match_rows], tracker_ids_t
                    [alpha_match_cols]] += 1
    for a, alpha in enumerate(array_labels):
        matches_count = matches_counts[a]
        matches_count / np.maximum(1, gt_id_count + tracker_id_count -
            matches_count)
        ass_re = matches_count / np.maximum(1, gt_id_count)
        res['AssRe'][a] = np.sum(matches_count * ass_re) / np.maximum(1,
            res['HOTA_TP'][a])
        ass_pr = matches_count / np.maximum(1, tracker_id_count)
        res['AssPr'][a] = np.sum(matches_count * ass_pr) / np.maximum(1,
            res['HOTA_TP'][a])
        res['AssA'][a] = res['AssRe'][a] * res['AssPr'][a] / np.maximum(
            1e-10, res['AssRe'][a] + res['AssPr'][a] - res['AssRe'][a] *
            res['AssPr'][a])
    num_attibutes_per_bbox = 5
    num_lacked_tracks = int((bboxes_track == -1.0).values.sum() /
        num_attibutes_per_bbox)
    res['HOTA_FP'] = res['HOTA_FP'] - num_lacked_tracks
    res['LocA'] = np.maximum(1e-10, res['LocA']) / np.maximum(1e-10, res[
        'HOTA_TP'])
    res['DetRe'] = res['HOTA_TP'] / np.maximum(1, res['HOTA_TP'] + res[
        'HOTA_FN'])
    res['DetPr'] = res['HOTA_TP'] / np.maximum(1, res['HOTA_TP'] + res[
        'HOTA_FP'])
    res['DetA'] = res['HOTA_TP'] / np.maximum(1, res['HOTA_TP'] + res[
        'HOTA_FN'] + res['HOTA_FP'])
    res['HOTA'] = np.sqrt(res['DetA'] * res['AssA'])
    res['RHOTA'] = np.sqrt(res['DetRe'] * res['AssA'])
    res['HOTA(0)'] = np.sqrt(res['DetA'] * res['AssA'])[0]
    res['LocA(0)'] = res['LocA'][0]
    res['HOTALocA(0)'] = res['HOTA(0)'] * res['LocA(0)']
    hota_final_scores(res)
    return res
