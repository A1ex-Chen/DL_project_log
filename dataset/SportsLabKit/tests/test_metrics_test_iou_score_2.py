def test_iou_score_2(self):
    bbox_det = [10, 10, 20, 20]
    bbox_gt = [10, 10, 20, 20]
    iou = iou_score(bbox_det, bbox_gt)
    ans = 1.0
    self.assertEqual(iou, ans)
