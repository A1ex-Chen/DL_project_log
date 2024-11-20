def test_ap_score_range_1(self):
    """Test average AP score within specified range with perfect detection."""
    bboxes_det = [[10, 10, 20, 20, 1.0, 'A', 'N'], [10, 10, 20, 20, 1.0,
        'A', 'M']]
    bboxes_gt = [[10, 10, 20, 20, 1, 'A', 'N'], [10, 10, 20, 20, 1, 'A', 'M']]
    ap_range = ap_score_range(bboxes_det, bboxes_gt, 0.5, 0.95, 0.05)
    ans = 1.0
    self.assertEqual(ap_range, ans)
