def evaluate(self, iou_threshold: float) ->Tuple[float, Dict[int, Result]]:
    sorted_indices = self.unfolded_pred_probs.argsort(axis=0)[::-1]
    sorted_unfolded_pred_image_ids = [self.unfolded_pred_image_ids[i] for i in
        sorted_indices]
    sorted_unfolded_pred_bboxes = self.unfolded_pred_bboxes[sorted_indices]
    sorted_unfolded_pred_classes = self.unfolded_pred_classes[sorted_indices]
    sorted_unfolded_pred_probs = self.unfolded_pred_probs[sorted_indices]
    class_to_result_dict = {}
    for c in range(1, self.num_classes):
        result = self._interpolated_average_precision(target_class=c,
            iou_threshold=iou_threshold, sorted_unfolded_image_ids=
            sorted_unfolded_pred_image_ids, sorted_unfolded_pred_bboxes=
            sorted_unfolded_pred_bboxes, sorted_unfolded_pred_classes=
            sorted_unfolded_pred_classes, sorted_unfolded_pred_probs=
            sorted_unfolded_pred_probs, image_id_to_gt_bboxes_dict=self.
            image_id_to_gt_bboxes_dict, image_id_to_gt_classes_dict=self.
            image_id_to_gt_classes_dict, image_id_to_gt_difficulties_dict=
            self.image_id_to_gt_difficulties_dict)
        class_to_result_dict[c] = result
    mean_ap = sum([result.ap for result in class_to_result_dict.values()]
        ) / len(class_to_result_dict)
    return mean_ap, class_to_result_dict
