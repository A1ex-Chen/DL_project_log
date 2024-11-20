def mota_score(bboxes_track: BBoxDataFrame, bboxes_gt: BBoxDataFrame) ->dict[
    str, Any]:
    """Calculates CLEAR metrics for one sequence.

    Args:
        bboxes_track (BBoxDataFrame): Bbox Dataframe for tracking in 1 sequence
        bboxes_gt (BBoxDataFrame): Bbox Dataframe for ground truth in 1 sequence

    Returns:
        dict[str, Any]: CLEAR metrics

    Note:
    The description of each evaluation indicator will be as follows:
    "MOTA"  :   Multi-Object Tracking Accuracy.
    "MOTAL" :   MOTA with a logarithmic penalty for ID switches.
    "MOTP"  :   The average dissimilarity between all true positives and their corresponding ground truth targets.
                res["MOTP_sum"] / np.maximum(1.0, res["CLR_TP"])
    "MODA"  :   Multi-Object Detection Accuracy. This measure combines false positives and missed targets.
    "CLR_Re":   MOTA's Recall. ["CLR_TP"] / np.maximum(1.0, res["CLR_TP"] + res["CLR_FN"]).
    "CLR_Pr":   MOTA's Precision. ["CLR_TP"] / np.maximum(1.0, res["CLR_TP"] + res["CLR_FP"]).
    "MTR"   :   MT divided by the number of unique IDs in gt.
    "PTR"   :   PT divided by the number of unique IDs in gt.
    "MLR"   :   ML divided by the number of unique IDs in gt.
    "sMOTA" :   Sum of similarity scores for matched bboxes.
    "CLR_TP" :  Number of TPs.
    "CLR_FN" :  Number of FNs.
    "CLR_FP" :  Number of FPs.
    "IDSW" :    Number of IDSW.
    "MT" :      Mostly tracked trajectory. A target is mostly tracked if it is successfully tracked for at least 80% of its life span.
    "PT" :      Partially tracked trajectory. All trajectories except MT and ML are PT.
    "ML" :      Mostly lost trajectory. If a track is only recovered for less than 20% of its total length, it is said to be mostly lost (ML).
    "Frag" :    Number of fragments. A fragment is a sub-trajectory of a track that is interrupted by a large gap in detection.

    This is also based on the following original paper and the github repository.
    paper : https://arxiv.org/pdf/1603.00831.pdf
    code  : https://github.com/JonathonLuiten/TrackEval
    """
    data = to_mot_eval_format(bboxes_gt, bboxes_track)
    main_integer_fields = ['CLR_TP', 'CLR_FN', 'CLR_FP', 'IDSW', 'MT', 'PT',
        'ML', 'Frag']
    main_float_fields = ['MOTA', 'MOTP', 'MODA', 'CLR_Re', 'CLR_Pr', 'MTR',
        'PTR', 'MLR', 'sMOTA']
    extra_integer_fields = ['CLR_Frames']
    integer_fields = main_integer_fields + extra_integer_fields
    extra_float_fields = ['CLR_F1', 'FP_per_frame', 'MOTAL', 'MOTP_sum']
    float_fields = main_float_fields + extra_float_fields
    fields = float_fields + integer_fields
    threshold = 0.5
    res = {}
    for field in fields:
        res[field] = 0
    if data['num_tracker_dets'] == 0:
        res['CLR_FN'] = data['num_gt_dets']
        res['ML'] = data['num_gt_ids']
        res['CLR_Frames'] = data['num_timesteps']
        res['MLR'] = 1
        mota_final_scores(res)
        return res
    if data['num_gt_dets'] == 0:
        res['CLR_FP'] = data['num_tracker_dets']
        res['CLR_Frames'] = data['num_timesteps']
        res['MLR'] = 1
        mota_final_scores(res)
        return res
    num_gt_ids = data['num_gt_ids']
    gt_id_count = np.zeros(num_gt_ids)
    gt_matched_count = np.zeros(num_gt_ids)
    gt_frag_count = np.zeros(num_gt_ids)
    prev_tracker_id = np.nan * np.zeros(num_gt_ids)
    prev_timestep_tracker_id = np.nan * np.zeros(num_gt_ids)
    for t, (gt_ids_t, tracker_ids_t) in enumerate(zip(data['gt_ids'], data[
        'tracker_ids'])):
        if len(gt_ids_t) == 0:
            res['CLR_FP'] += len(tracker_ids_t)
            continue
        if len(tracker_ids_t) == 0:
            res['CLR_FN'] += len(gt_ids_t)
            gt_id_count[gt_ids_t] += 1
            continue
        similarity = data['similarity_scores'][t]
        score_mat = tracker_ids_t[np.newaxis, :] == prev_timestep_tracker_id[
            gt_ids_t[:, np.newaxis]]
        score_mat = 1000 * score_mat + similarity
        score_mat[similarity < threshold - np.finfo('float').eps] = 0
        match_rows, match_cols = linear_sum_assignment(-score_mat)
        actually_matched_mask = score_mat[match_rows, match_cols
            ] > 0 + np.finfo('float').eps
        match_rows = match_rows[actually_matched_mask]
        match_cols = match_cols[actually_matched_mask]
        matched_gt_ids = gt_ids_t[match_rows]
        matched_tracker_ids = tracker_ids_t[match_cols]
        prev_matched_tracker_ids = prev_tracker_id[matched_gt_ids]
        is_idsw = np.logical_not(np.isnan(prev_matched_tracker_ids)
            ) & np.not_equal(matched_tracker_ids, prev_matched_tracker_ids)
        res['IDSW'] += np.sum(is_idsw)
        gt_id_count[gt_ids_t] += 1
        gt_matched_count[matched_gt_ids] += 1
        not_previously_tracked = np.isnan(prev_timestep_tracker_id)
        prev_tracker_id[matched_gt_ids] = matched_tracker_ids
        prev_timestep_tracker_id[:] = np.nan
        prev_timestep_tracker_id[matched_gt_ids] = matched_tracker_ids
        currently_tracked = np.logical_not(np.isnan(prev_timestep_tracker_id))
        gt_frag_count += np.logical_and(not_previously_tracked,
            currently_tracked)
        num_matches = len(matched_gt_ids)
        res['CLR_TP'] += num_matches
        res['CLR_FN'] += len(gt_ids_t) - num_matches
        res['CLR_FP'] += len(tracker_ids_t) - num_matches
        if num_matches > 0:
            res['MOTP_sum'] += sum(similarity[match_rows, match_cols])
        tracked_ratio = gt_matched_count[gt_id_count > 0] / gt_id_count[
            gt_id_count > 0]
        res['MT'] = np.sum(np.greater(tracked_ratio, 0.8))
        res['PT'] = np.sum(np.greater_equal(tracked_ratio, 0.2)) - res['MT']
        res['ML'] = num_gt_ids - res['MT'] - res['PT']
        res['Frag'] = np.sum(np.subtract(gt_frag_count[gt_frag_count > 0], 1))
        res['MOTP'] = res['MOTP_sum'] / np.maximum(1.0, res['CLR_TP'])
        res['CLR_Frames'] = data['num_timesteps']
    num_attibutes_per_bbox = 5
    num_lacked_tracks = int((bboxes_track == -1.0).values.sum() /
        num_attibutes_per_bbox)
    res['CLR_FP'] = res['CLR_FP'] - num_lacked_tracks
    mota_final_scores(res)
    return res
