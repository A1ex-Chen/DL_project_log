def test_iou_score_3(self):
    bbox_det = [10, 10, 20, 20]
    bbox_gt = [0, 1, 2, 3]
    iou = iou_score(bbox_det, bbox_gt)
    ans = 0
    self.assertEqual(iou, ans)
