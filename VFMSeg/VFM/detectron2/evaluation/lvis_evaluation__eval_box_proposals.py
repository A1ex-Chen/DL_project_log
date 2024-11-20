def _eval_box_proposals(self, predictions):
    """
        Evaluate the box proposals in predictions.
        Fill self._results with the metrics for "box_proposals" task.
        """
    if self._output_dir:
        bbox_mode = BoxMode.XYXY_ABS.value
        ids, boxes, objectness_logits = [], [], []
        for prediction in predictions:
            ids.append(prediction['image_id'])
            boxes.append(prediction['proposals'].proposal_boxes.tensor.numpy())
            objectness_logits.append(prediction['proposals'].
                objectness_logits.numpy())
        proposal_data = {'boxes': boxes, 'objectness_logits':
            objectness_logits, 'ids': ids, 'bbox_mode': bbox_mode}
        with PathManager.open(os.path.join(self._output_dir,
            'box_proposals.pkl'), 'wb') as f:
            pickle.dump(proposal_data, f)
    if not self._do_evaluation:
        self._logger.info('Annotations are not available for evaluation.')
        return
    self._logger.info('Evaluating bbox proposals ...')
    res = {}
    areas = {'all': '', 'small': 's', 'medium': 'm', 'large': 'l'}
    for limit in [100, 1000]:
        for area, suffix in areas.items():
            stats = _evaluate_box_proposals(predictions, self._lvis_api,
                area=area, limit=limit)
            key = 'AR{}@{:d}'.format(suffix, limit)
            res[key] = float(stats['ar'].item() * 100)
    self._logger.info('Proposal metrics: \n' + create_small_table(res))
    self._results['box_proposals'] = res
