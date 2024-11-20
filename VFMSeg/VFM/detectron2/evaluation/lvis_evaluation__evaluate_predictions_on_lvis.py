def _evaluate_predictions_on_lvis(lvis_gt, lvis_results, iou_type,
    max_dets_per_image=None, class_names=None):
    """
    Args:
        iou_type (str):
        max_dets_per_image (None or int): limit on maximum detections per image in evaluating AP
            This limit, by default of the LVIS dataset, is 300.
        class_names (None or list[str]): if provided, will use it to predict
            per-category AP.

    Returns:
        a dict of {metric name: score}
    """
    metrics = {'bbox': ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'APr',
        'APc', 'APf'], 'segm': ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl',
        'APr', 'APc', 'APf']}[iou_type]
    logger = logging.getLogger(__name__)
    if len(lvis_results) == 0:
        logger.warn('No predictions from the model!')
        return {metric: float('nan') for metric in metrics}
    if iou_type == 'segm':
        lvis_results = copy.deepcopy(lvis_results)
        for c in lvis_results:
            c.pop('bbox', None)
    if max_dets_per_image is None:
        max_dets_per_image = 300
    from lvis import LVISEval, LVISResults
    logger.info(
        f'Evaluating with max detections per image = {max_dets_per_image}')
    lvis_results = LVISResults(lvis_gt, lvis_results, max_dets=
        max_dets_per_image)
    lvis_eval = LVISEval(lvis_gt, lvis_results, iou_type)
    lvis_eval.run()
    lvis_eval.print_results()
    results = lvis_eval.get_results()
    results = {metric: float(results[metric] * 100) for metric in metrics}
    logger.info('Evaluation results for {}: \n'.format(iou_type) +
        create_small_table(results))
    return results
