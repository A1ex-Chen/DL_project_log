def _evaluate(self, prediction: Prediction) ->Evaluation:
    accuracy = metrics.accuracy_score(y_true=prediction.
        sorted_all_gt_classes, y_pred=prediction.sorted_all_pred_classes).item(
        )
    avg_recall = metrics.recall_score(y_true=prediction.
        sorted_all_gt_classes, y_pred=prediction.sorted_all_pred_classes,
        average='macro').item()
    avg_precision = metrics.precision_score(y_true=prediction.
        sorted_all_gt_classes, y_pred=prediction.sorted_all_pred_classes,
        average='macro').item()
    avg_f1_score = metrics.f1_score(y_true=prediction.sorted_all_gt_classes,
        y_pred=prediction.sorted_all_pred_classes, average='macro').item()
    confusion_matrix = metrics.confusion_matrix(y_true=prediction.
        sorted_all_gt_classes, y_pred=prediction.sorted_all_pred_classes)
    class_to_fpr_array_dict = {}
    class_to_tpr_array_dict = {}
    class_to_thresh_array_dict = {}
    class_to_auc_dict = {}
    class_to_sensitivity_dict = {}
    class_to_specificity_dict = {}
    for c in range(1, self._num_classes):
        c_gt_classes = prediction.sorted_all_gt_classes == c
        c_pred_classes = prediction.sorted_all_pred_classes == c
        fpr_array, tpr_array, thresh_array = metrics.roc_curve(y_true=
            c_gt_classes, y_score=prediction.sorted_all_pred_probs *
            c_pred_classes.float())
        class_to_fpr_array_dict[c] = fpr_array
        class_to_tpr_array_dict[c] = tpr_array
        class_to_thresh_array_dict[c] = thresh_array
        auc = metrics.auc(fpr_array, tpr_array)
        class_to_auc_dict[c] = float(auc)
        num_tps = (c_gt_classes & c_pred_classes).sum().item()
        num_tns = (~c_gt_classes & ~c_pred_classes).sum().item()
        num_fps = (~c_gt_classes & c_pred_classes).sum().item()
        num_fns = (c_gt_classes & ~c_pred_classes).sum().item()
        sensitivity = num_tps / (num_tps + num_fns + np.finfo(np.float32).eps)
        specificity = num_tns / (num_tns + num_fps + np.finfo(np.float32).eps)
        class_to_sensitivity_dict[c] = float(sensitivity)
        class_to_specificity_dict[c] = float(specificity)
    mean_auc = sum([v for _, v in class_to_auc_dict.items()]) / len(
        class_to_auc_dict)
    mean_sensitivity = sum([v for _, v in class_to_sensitivity_dict.items()]
        ) / len(class_to_sensitivity_dict)
    mean_specificity = sum([v for _, v in class_to_specificity_dict.items()]
        ) / len(class_to_specificity_dict)
    evaluation = Evaluator.Evaluation(accuracy, avg_recall, avg_precision,
        avg_f1_score, confusion_matrix, class_to_fpr_array_dict,
        class_to_tpr_array_dict, class_to_thresh_array_dict, metric_auc=
        Evaluator.Evaluation.MetricResult(mean_auc, class_to_auc_dict),
        metric_sensitivity=Evaluator.Evaluation.MetricResult(
        mean_sensitivity, class_to_sensitivity_dict), metric_specificity=
        Evaluator.Evaluation.MetricResult(mean_specificity,
        class_to_specificity_dict))
    return evaluation
