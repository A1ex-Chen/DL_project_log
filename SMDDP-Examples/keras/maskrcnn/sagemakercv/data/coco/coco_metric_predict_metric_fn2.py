def predict_metric_fn2(self, predictions, source_ids, is_predict_image_mask
    =False, groundtruth_data=None):
    """Generates COCO metrics."""
    image_ids = list(set(source_ids))
    if groundtruth_data is not None:
        self.coco_gt.reset(groundtruth_data)
    coco_dt = self.coco_gt.loadRes2(predictions, self._include_mask,
        is_image_mask=is_predict_image_mask)
    coco_eval = COCOeval(self.coco_gt, coco_dt, iouType='bbox')
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    coco_metrics = coco_eval.stats
    if self._include_mask:
        mcoco_eval = COCOeval(self.coco_gt, coco_dt, iouType='segm')
        mcoco_eval.params.imgIds = image_ids
        mcoco_eval.evaluate()
        mcoco_eval.accumulate()
        mcoco_eval.summarize()
        mask_coco_metrics = mcoco_eval.stats
    if self._include_mask:
        metrics = np.hstack((coco_metrics, mask_coco_metrics))
    else:
        metrics = coco_metrics
    self._reset()
    metrics = metrics.astype(np.float32)
    metrics_dict = {}
    for i, name in enumerate(self.metric_names):
        metrics_dict[name] = metrics[i]
    return metrics_dict
