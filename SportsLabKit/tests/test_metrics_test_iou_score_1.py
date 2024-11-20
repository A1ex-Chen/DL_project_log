def test_iou_score_1(self):
    bbox_det = [10, 10, 20, 20]
    bbox_gt = [10, 15, 20, 20]
    iou = iou_score(bbox_det, bbox_gt)
    ans = 0.5
    self.assertEqual(iou, ans)
