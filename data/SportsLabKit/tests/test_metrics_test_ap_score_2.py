def test_ap_score_2(self):
    """Test AP score with zero detection."""
    bboxes_det = []
    bboxes_gt = [[10, 10, 20, 20, 1, 'A', 'N'], [10, 10, 20, 20, 1, 'A', 'M']]
    ap = ap_score(bboxes_det, bboxes_gt, 0.5)
    ans = {'class': 'A', 'precision': [], 'recall': [], 'AP': 0.0,
        'interpolated precision': [], 'interpolated recall': [],
        'total positives': 0, 'total TP': 0, 'total FP': 0}
    self.assertDictEqual(ap, ans)
