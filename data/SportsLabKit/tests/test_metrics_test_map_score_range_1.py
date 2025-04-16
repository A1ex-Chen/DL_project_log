def test_map_score_range_1(self):
    """Test average mAP score within specified range with perfect detection using a list of 7 values as input"""
    bboxes_det = [[10, 10, 20, 20, 1.0, 'A', 'N'], [10, 10, 20, 20, 1.0,
        'B', 'M']]
    bboxes_gt = [[10, 10, 20, 20, 1, 'A', 'N'], [10, 10, 20, 20, 1, 'B', 'M']]
    mAP_range = map_score_range(bboxes_det, bboxes_gt, 0.5, 0.95, 0.05)
    ans = 1
    self.assertEqual(mAP_range, ans)
