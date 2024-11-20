def test_hota_score_1(self):
    """Test HOTA score with perfect detection."""
    bboxes_track = bbdf
    bboxes_gt = bbdf
    hota = hota_score(bboxes_track, bboxes_gt)
    HOTA_TP = 46.0
    HOTA_FN = 0.0
    HOTA_FP = 0.0
    AssA = 1.0
    AssRe = 1.0
    AssPr = 1.0
    LocA = 1.0
    DetA = HOTA_TP / np.maximum(1.0, HOTA_TP + HOTA_FN + HOTA_FP)
    DetRe = HOTA_TP / np.maximum(1.0, HOTA_TP + HOTA_FN)
    DetPr = HOTA_TP / np.maximum(1.0, HOTA_TP + HOTA_FP)
    HOTA = np.sqrt(DetA * AssA)
    RHOTA = np.sqrt(DetA * AssA)
    HOTA0 = np.sqrt(DetA * AssA)
    LocA0 = 1.0
    HOTALocA0 = np.sqrt(DetA * AssA) * LocA0
    ans = {'HOTA': HOTA, 'DetA': DetA, 'AssA': AssA, 'DetRe': DetRe,
        'DetPr': DetPr, 'AssRe': AssRe, 'AssPr': AssPr, 'LocA': LocA,
        'RHOTA': RHOTA, 'HOTA_TP': HOTA_TP, 'HOTA_FN': HOTA_FN, 'HOTA_FP':
        HOTA_FP, 'HOTA(0)': HOTA0, 'LocA(0)': LocA0, 'HOTALocA(0)': HOTALocA0}
    self.assertDictEqual(hota, ans)
