def _evaluate_predictions_on_coco(self, coco_gt, coco_results):
    """
        Evaluate the coco results using COCOEval API.
        """
    assert len(coco_results) > 0
    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = RotatedCOCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval
