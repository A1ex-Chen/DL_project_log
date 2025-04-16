def _derive_refcoco_results(self, coco_eval, iou_type):
    """
        Derive the desired score numbers from summarized COCOeval.

        Args:
            coco_eval (None or COCOEval): None represents no predictions from model.
            iou_type (str):
            class_names (None or list[str]): if provided, will use it to predict
                per-category AP.

        Returns:
            a dict of {metric name: score}
        """
    metrics = {'bbox': ['P@0.5', 'P@0.6', 'P@0.7', 'P@0.8', 'P@0.9'],
        'segm': ['oIoU', 'mIoU']}[iou_type]
    if coco_eval is None:
        self._logger.warn('No predictions from the model!')
        return {metric: float('nan') for metric in metrics}
    results = {metric: float('nan') for idx, metric in enumerate(metrics)}
    ious = np.array([v for k, v in coco_eval.ious.items()])
    total_intersection_area = coco_eval.total_intersection_area
    total_union_area = coco_eval.total_union_area
    iou_list = coco_eval.iou_list
    if iou_type == 'bbox':
        results['P@0.5'] = np.sum(ious > 0.5) / len(ious) * 100
        results['P@0.6'] = np.sum(ious > 0.6) / len(ious) * 100
        results['P@0.7'] = np.sum(ious > 0.7) / len(ious) * 100
        results['P@0.8'] = np.sum(ious > 0.8) / len(ious) * 100
        results['P@0.9'] = np.sum(ious > 0.9) / len(ious) * 100
    elif iou_type == 'segm':
        results['oIoU'] = total_intersection_area / total_union_area * 100
        results['mIoU'] = np.mean(ious) * 100
    else:
        raise ValueError('Unsupported iou_type!')
    self._logger.info('Evaluation results for {}: \n'.format(iou_type) +
        create_small_table(results))
    return results
