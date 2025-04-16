def test_ap_score_1(self):
    """Test AP score with perfect detection."""
    bboxes_det = [[10, 10, 20, 20, 1.0, 'A', 'N'], [10, 10, 20, 20, 1.0,
        'A', 'M']]
    bboxes_gt = [[10, 10, 20, 20, 1, 'A', 'N'], [10, 10, 20, 20, 1, 'A', 'M']]
    ap = ap_score(bboxes_det, bboxes_gt, 0.5)
    ans = {'class': 'A', 'precision': [1.0, 1.0], 'recall': [0.5, 1.0],
        'AP': 1.0, 'interpolated precision': [(1.0) for i in range(11)],
        'interpolated recall': list(np.linspace(0, 1, 11)[::-1]),
        'total positives': 2, 'total TP': 2, 'total FP': 0}
    self.assertDictEqual(ap, ans)
