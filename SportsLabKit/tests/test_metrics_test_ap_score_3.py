def test_ap_score_3(self):
    """Test AP score with false negative detection."""
    bboxes_det = [[10, 10, 20, 20, 1.0, 'A', 'N'], [10, 10, 20, 20, 1.0,
        'A', 'M']]
    bboxes_gt = []
    with self.assertRaises(AssertionError):
        ap_score(bboxes_det, bboxes_gt, 0.5)
