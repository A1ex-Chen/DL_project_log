def _evaluate_with_condition(self, prediction: Prediction, quality:
    Evaluation.Quality, size: Evaluation.Size, returns_coco_result: bool
    ) ->Evaluation:
    assert size == Evaluator.Evaluation.Size.ALL, 'Only `Size.ALL` is supported now'
    metric = AveragePrecision({k: v.numpy() for k, v in prediction.
        image_id_to_pred_bboxes_dict.items()}, {k: v.numpy() for k, v in
        prediction.image_id_to_pred_classes_dict.items()}, {k: v.numpy() for
        k, v in prediction.image_id_to_pred_probs_dict.items()}, {k: v.
        numpy() for k, v in prediction.image_id_to_gt_bboxes_dict.items()},
        {k: v.numpy() for k, v in prediction.image_id_to_gt_classes_dict.
        items()}, {k: v.numpy() for k, v in prediction.
        image_id_to_difficulties_dict.items()}, self._num_classes)
    mean_ap, class_to_result_dict = metric.evaluate(iou_threshold=quality.
        to_iou_threshold())
    class_to_ap_dict = {}
    class_to_inter_recall_array_dict = {}
    class_to_inter_precision_array_dict = {}
    class_to_recall_array_dict = {}
    class_to_precision_array_dict = {}
    class_to_accuracy_array_dict = {}
    class_to_f1_score_array_dict = {}
    class_to_prob_array_dict = {}
    class_to_top_f1_score_dict = {}
    class_to_recall_at_top_f1_score_dict = {}
    class_to_precision_at_top_f1_score_dict = {}
    class_to_accuracy_at_top_f1_score_dict = {}
    for c in range(1, self._num_classes):
        result = class_to_result_dict[c]
        ap = result.ap
        inter_recall_array = result.inter_recall_array
        inter_precision_array = result.inter_precision_array
        recall_array = result.recall_array
        precision_array = result.precision_array
        accuracy_array = result.accuracy_array
        f1_score_array = 2 * recall_array * precision_array / (recall_array +
            precision_array + np.finfo(np.float32).eps)
        prob_array = result.prob_array
        class_to_ap_dict[c] = ap
        class_to_inter_recall_array_dict[c] = inter_recall_array
        class_to_inter_precision_array_dict[c] = inter_precision_array
        class_to_recall_array_dict[c] = recall_array
        class_to_precision_array_dict[c] = precision_array
        class_to_accuracy_array_dict[c] = accuracy_array
        class_to_f1_score_array_dict[c] = f1_score_array
        class_to_prob_array_dict[c] = prob_array
        if f1_score_array.shape[0] > 0:
            top_f1_score_index = f1_score_array.argmax().item()
            class_to_top_f1_score_dict[c] = f1_score_array.max().item()
            class_to_recall_at_top_f1_score_dict[c] = recall_array[
                top_f1_score_index].item()
            class_to_precision_at_top_f1_score_dict[c] = precision_array[
                top_f1_score_index].item()
            class_to_accuracy_at_top_f1_score_dict[c] = accuracy_array[
                top_f1_score_index].item()
        else:
            class_to_top_f1_score_dict[c] = 0.0
            class_to_recall_at_top_f1_score_dict[c] = 0.0
            class_to_precision_at_top_f1_score_dict[c] = 0.0
            class_to_accuracy_at_top_f1_score_dict[c] = 0.0
    mean_top_f1_score = sum([v for _, v in class_to_top_f1_score_dict.items()]
        ) / len(class_to_top_f1_score_dict)
    mean_recall_at_top_f1_score = sum([v for _, v in
        class_to_recall_at_top_f1_score_dict.items()]) / len(
        class_to_recall_at_top_f1_score_dict)
    mean_precision_at_top_f1_score = sum([v for _, v in
        class_to_precision_at_top_f1_score_dict.items()]) / len(
        class_to_precision_at_top_f1_score_dict)
    mean_accuracy_at_top_f1_score = sum([v for _, v in
        class_to_accuracy_at_top_f1_score_dict.items()]) / len(
        class_to_accuracy_at_top_f1_score_dict)
    coco_result = metric.evaluate_by_pycocotools(
        ) if returns_coco_result else None
    evaluation = Evaluator.Evaluation(quality, size,
        class_to_inter_recall_array_dict,
        class_to_inter_precision_array_dict, class_to_recall_array_dict,
        class_to_precision_array_dict, class_to_accuracy_array_dict,
        class_to_f1_score_array_dict, class_to_prob_array_dict, metric_ap=
        Evaluator.Evaluation.MetricResult(mean_ap, class_to_ap_dict),
        metric_top_f1_score=Evaluator.Evaluation.MetricResult(
        mean_top_f1_score, class_to_top_f1_score_dict),
        metric_recall_at_top_f1_score=Evaluator.Evaluation.MetricResult(
        mean_recall_at_top_f1_score, class_to_recall_at_top_f1_score_dict),
        metric_precision_at_top_f1_score=Evaluator.Evaluation.MetricResult(
        mean_precision_at_top_f1_score,
        class_to_precision_at_top_f1_score_dict),
        metric_accuracy_at_top_f1_score=Evaluator.Evaluation.MetricResult(
        mean_accuracy_at_top_f1_score,
        class_to_accuracy_at_top_f1_score_dict), coco_result=coco_result)
    return evaluation
