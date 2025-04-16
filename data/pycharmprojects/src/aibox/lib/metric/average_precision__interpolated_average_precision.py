@staticmethod
def _interpolated_average_precision(target_class: int, iou_threshold: float,
    sorted_unfolded_image_ids: List[str], sorted_unfolded_pred_bboxes: np.
    ndarray, sorted_unfolded_pred_classes: np.ndarray,
    sorted_unfolded_pred_probs: np.ndarray, image_id_to_gt_bboxes_dict:
    Dict[str, np.ndarray], image_id_to_gt_classes_dict: Dict[str, np.
    ndarray], image_id_to_gt_difficulties_dict: Dict[str, np.ndarray]
    ) ->Result:
    image_id_to_detected_gt_indices_dict = defaultdict(list)
    num_tps_array, num_fps_array = [], []
    prob_array = []
    target_indices = (sorted_unfolded_pred_classes == target_class).nonzero()[0
        ].tolist()
    for idx in target_indices:
        image_id = sorted_unfolded_image_ids[idx]
        pred_bbox = sorted_unfolded_pred_bboxes[idx]
        pred_prob = sorted_unfolded_pred_probs[idx]
        gt_bboxes = image_id_to_gt_bboxes_dict[image_id]
        gt_classes = image_id_to_gt_classes_dict[image_id]
        gt_difficulties = image_id_to_gt_difficulties_dict[image_id]
        detected_gt_indices = image_id_to_detected_gt_indices_dict[image_id]
        c_gt_bboxes = gt_bboxes[gt_classes == target_class]
        c_gt_difficulties = gt_difficulties[gt_classes == target_class]
        if c_gt_bboxes.shape[0] == 0:
            num_tps_array.append(0)
            num_fps_array.append(1)
            prob_array.append(pred_prob)
            continue
        coco_c_gt_bboxes = c_gt_bboxes.copy()
        coco_c_gt_bboxes[:, [2, 3]] -= coco_c_gt_bboxes[:, [0, 1]]
        coco_pred_bboxes = pred_bbox[None, :].copy()
        coco_pred_bboxes[:, [2, 3]] -= coco_pred_bboxes[:, [0, 1]]
        pred_to_gts_iou = maskUtils.iou(coco_pred_bboxes.tolist(),
            coco_c_gt_bboxes.tolist(), c_gt_difficulties.tolist())[0]
        assert pred_to_gts_iou.shape == (c_gt_bboxes.shape[0],)
        matched_gt_indices = ((c_gt_difficulties == 0) * (pred_to_gts_iou >
            iou_threshold)).nonzero()[0]
        if matched_gt_indices.shape[0] > 0:
            non_detected_matched_gt_indices = np.array([i for i in
                matched_gt_indices if i not in detected_gt_indices])
            if non_detected_matched_gt_indices.shape[0] > 0:
                pred_to_gt_max_index = pred_to_gts_iou[
                    non_detected_matched_gt_indices].argmax(axis=0)
                pred_to_gt_max_index = non_detected_matched_gt_indices[
                    pred_to_gt_max_index]
            else:
                pred_to_gt_max_index = pred_to_gts_iou[matched_gt_indices
                    ].argmax(axis=0)
                pred_to_gt_max_index = matched_gt_indices[pred_to_gt_max_index]
        else:
            pred_to_gt_max_index = pred_to_gts_iou.argmax(axis=0)
        pred_to_gt_max_index = pred_to_gt_max_index.item()
        pred_to_gt_max_iou = pred_to_gts_iou[pred_to_gt_max_index].item()
        if pred_to_gt_max_iou > iou_threshold:
            if not c_gt_difficulties[pred_to_gt_max_index]:
                if pred_to_gt_max_index not in detected_gt_indices:
                    num_tps_array.append(1)
                    num_fps_array.append(0)
                    prob_array.append(pred_prob)
                    detected_gt_indices.append(pred_to_gt_max_index)
                else:
                    num_tps_array.append(0)
                    num_fps_array.append(1)
                    prob_array.append(pred_prob)
            else:
                num_tps_array.append(0)
                num_fps_array.append(0)
                prob_array.append(pred_prob)
        else:
            num_tps_array.append(0)
            num_fps_array.append(1)
            prob_array.append(pred_prob)
    num_tps_array = np.array(num_tps_array, np.float).cumsum()
    num_fps_array = np.array(num_fps_array, np.float).cumsum()
    prob_array = np.array(prob_array, np.float)
    num_positives = 0
    for gt_classes, gt_difficulties in zip(image_id_to_gt_classes_dict.
        values(), image_id_to_gt_difficulties_dict.values()):
        class_mask = gt_classes == target_class
        num_positives += class_mask.sum().item()
        num_positives -= (gt_difficulties[class_mask] == 1).sum().item()
    recall_array: np.ndarray = num_tps_array / np.maximum(num_positives, np
        .finfo(np.float32).eps)
    precision_array: np.ndarray = num_tps_array / np.maximum(num_tps_array +
        num_fps_array, np.finfo(np.float32).eps)
    accuracy_array: np.ndarray = num_tps_array / np.maximum(num_positives +
        num_fps_array, np.finfo(np.float32).eps)
    recall_and_interpolated_precision_list = []
    for r in np.arange(0.0, 1.01, 0.01):
        if np.sum(recall_array >= r) == 0:
            p = 0
        else:
            p = np.max(precision_array[recall_array >= r]).item()
        recall_and_interpolated_precision_list.append((r, p))
    ap: float = np.mean([p for r, p in recall_and_interpolated_precision_list]
        ).item()
    inter_recall_array = np.array([r for r, p in
        recall_and_interpolated_precision_list])
    inter_precision_array = np.array([p for r, p in
        recall_and_interpolated_precision_list])
    return AveragePrecision.Result(ap, inter_recall_array,
        inter_precision_array, recall_array, precision_array,
        accuracy_array, prob_array)
