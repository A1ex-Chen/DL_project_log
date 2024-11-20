def test_hota_score_3(self):
    """Test HOTA score with half detection."""
    bboxes_track = player_dfs[0]
    bboxes_gt = pd.concat([player_dfs[0], player_dfs[1]], axis=1)
    hota = hota_score(bboxes_track, bboxes_gt)
    HOTA_TP = 2.0
    HOTA_FN = 2.0
    HOTA_FP = 0.0
    AssA = 1.0
    AssRe = 1.0
    AssPr = 1.0
    LocA = 1.0
    DetA = HOTA_TP / (HOTA_TP + HOTA_FN + HOTA_FP)
    DetRe = HOTA_TP / (HOTA_TP + HOTA_FN)
    DetPr = HOTA_TP / (HOTA_TP + HOTA_FP)
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
