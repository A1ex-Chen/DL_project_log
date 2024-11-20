def mota_final_scores(res):
    """Calculate final CLEAR scores"""
    num_gt_ids = res['MT'] + res['ML'] + res['PT']
    res['MTR'] = res['MT'] / np.maximum(1.0, num_gt_ids)
    res['MLR'] = res['ML'] / np.maximum(1.0, num_gt_ids)
    res['PTR'] = res['PT'] / np.maximum(1.0, num_gt_ids)
    res['CLR_Re'] = res['CLR_TP'] / np.maximum(1.0, res['CLR_TP'] + res[
        'CLR_FN'])
    res['CLR_Pr'] = res['CLR_TP'] / np.maximum(1.0, res['CLR_TP'] + res[
        'CLR_FP'])
    res['MODA'] = (res['CLR_TP'] - res['CLR_FP']) / np.maximum(1.0, res[
        'CLR_TP'] + res['CLR_FN'])
    res['MOTA'] = (res['CLR_TP'] - res['CLR_FP'] - res['IDSW']) / np.maximum(
        1.0, res['CLR_TP'] + res['CLR_FN'])
    res['MOTP'] = res['MOTP_sum'] / np.maximum(1.0, res['CLR_TP'])
    res['sMOTA'] = (res['MOTP_sum'] - res['CLR_FP'] - res['IDSW']
        ) / np.maximum(1.0, res['CLR_TP'] + res['CLR_FN'])
    res['CLR_F1'] = res['CLR_TP'] / np.maximum(1.0, res['CLR_TP'] + 0.5 *
        res['CLR_FN'] + 0.5 * res['CLR_FP'])
    res['FP_per_frame'] = res['CLR_FP'] / np.maximum(1.0, res['CLR_Frames'])
    safe_log_idsw = np.log10(res['IDSW']) if res['IDSW'] > 0 else res['IDSW']
    res['MOTAL'] = (res['CLR_TP'] - res['CLR_FP'] - safe_log_idsw
        ) / np.maximum(1.0, res['CLR_TP'] + res['CLR_FN'])
