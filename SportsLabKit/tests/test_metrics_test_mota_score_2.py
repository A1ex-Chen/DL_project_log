def test_mota_score_2(self):
    """Test MOTA score with zero tracking."""
    bboxes_track = bbdf.iloc[0:0]
    bboxes_gt = bbdf
    mota = mota_score(bboxes_track, bboxes_gt)
    CLR_TP = 0
    CLR_FN = 23 * 2
    CLR_FP = 0
    IDSW = 0
    FP_per_frame = 0.0
    MT = 0
    PT = 0
    ML = 23
    Frag = 0
    MOTP_sum = 0.0
    MTR = MT / 23
    PTR = PT / 23
    MLR = ML / 23
    sMOTA = (MOTP_sum - CLR_FP - IDSW) / np.maximum(1.0, CLR_TP + CLR_FN)
    MOTP = MOTP_sum / np.maximum(1.0, CLR_TP)
    CLR_Re = CLR_TP / np.maximum(1.0, CLR_TP + CLR_FN)
    CLR_Pr = CLR_TP / np.maximum(1.0, CLR_TP + CLR_FP)
    CLR_F1 = 2 * (CLR_Re * CLR_Pr) / np.maximum(1.0, CLR_Re + CLR_Pr)
    MOTA = 1 - (CLR_FN + CLR_FP + IDSW) / np.maximum(1.0, CLR_TP + CLR_FN)
    MODA = CLR_TP / np.maximum(1.0, CLR_TP + CLR_FN)
    safe_log_idsw = np.log10(IDSW) if IDSW > 0 else IDSW
    MOTAL = (CLR_TP - CLR_FP - safe_log_idsw) / np.maximum(1.0, CLR_TP + CLR_FN
        )
    ans = {'MOTA': MOTA, 'MOTP': MOTP, 'MODA': MODA, 'CLR_Re': CLR_Re,
        'CLR_Pr': CLR_Pr, 'MTR': MTR, 'PTR': PTR, 'MLR': MLR, 'sMOTA':
        sMOTA, 'CLR_F1': CLR_F1, 'FP_per_frame': FP_per_frame, 'MOTAL':
        MOTAL, 'MOTP_sum': MOTP_sum, 'CLR_TP': CLR_TP, 'CLR_FN': CLR_FN,
        'CLR_FP': CLR_FP, 'IDSW': IDSW, 'MT': MT, 'PT': PT, 'ML': ML,
        'Frag': Frag, 'CLR_Frames': len(bboxes_gt)}
    self.assertDictEqual(mota, ans)
