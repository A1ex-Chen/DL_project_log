def test_map_score_1(self):
    """Test mAP score with perfect detection using a list of 7 values as input"""
    bboxes_det = [[10, 10, 20, 20, 1.0, 'A', 'N'], [10, 10, 20, 20, 1.0,
        'B', 'M']]
    bboxes_gt = [[10, 10, 20, 20, 1, 'A', 'N'], [10, 10, 20, 20, 1, 'B', 'M']]
    mAP = map_score(bboxes_det, bboxes_gt, 0.5)
    ans = 1
    self.assertEqual(mAP, ans)
