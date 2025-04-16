def _evaluate_with_condition(self, prediction: Prediction, quality:
    Evaluation.Quality, size: Evaluation.Size, pred_needs_inv_process: bool
    ) ->Evaluation:
    assert quality in [Evaluator.Evaluation.Quality.STANDARD, Evaluator.
        Evaluation.Quality.STRICT, Evaluator.Evaluation.Quality.STRICTEST
        ], 'Only `Quality.STANDARD`, `Quality.STRICT` and `Quality.STRICTEST` are supported now'
    assert size in [Evaluator.Evaluation.Size.ALL, Evaluator.Evaluation.
        Size.AREA_L, Evaluator.Evaluation.Size.AREA_M, Evaluator.Evaluation
        .Size.AREA_S
        ], 'Only `Size.ALL`, `Size.AREA_L`, `Size.AREA_M` and `Size.AREA_S` are supported now'
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)

    def convert_to_coco_api(ds):
        coco_ds = COCO()
        ann_id = 1
        dataset = {'images': [], 'categories': [], 'annotations': []}
        categories = set()
        for img_idx in range(len(ds)):
            item = ds[img_idx]
            image_id = item.image_id
            img_dict = {}
            img_dict['id'] = image_id
            img_dict['height'] = item.image.shape[1]
            img_dict['width'] = item.image.shape[2]
            dataset['images'].append(img_dict)
            bboxes = item.bboxes
            bboxes[:, 2:] -= bboxes[:, :2]
            bboxes = bboxes.tolist()
            labels = item.classes.tolist()
            areas = item.bboxes[:, 2] * item.bboxes[:, 3]
            areas = areas.tolist()
            iscrowd = item.difficulties.tolist()
            masks = item.masks
            masks = masks.permute(0, 2, 1).contiguous().permute(0, 2, 1)
            num_objs = len(bboxes)
            for i in range(num_objs):
                ann = {}
                ann['image_id'] = image_id
                ann['bbox'] = bboxes[i]
                ann['category_id'] = labels[i]
                categories.add(labels[i])
                ann['area'] = areas[i]
                ann['iscrowd'] = iscrowd[i]
                ann['id'] = ann_id
                ann['segmentation'] = coco_mask.encode(masks[i].numpy().
                    astype(np.uint8))
                dataset['annotations'].append(ann)
                ann_id += 1
        dataset['categories'] = [{'id': i} for i in sorted(categories)]
        coco_ds.dataset = dataset
        coco_ds.createIndex()
        return coco_ds
    coco = convert_to_coco_api(self._dataset)
    iou_types = ['bbox', 'segm']
    coco_evaluator = CocoEvaluator(coco, iou_types)
    for image_id in prediction.image_id_to_pred_bboxes_dict.keys():
        output = {'boxes': Preprocessor.inv_process_bboxes(prediction.
            image_id_to_process_dict_dict[image_id], prediction.
            image_id_to_pred_bboxes_dict[image_id]) if
            pred_needs_inv_process else prediction.
            image_id_to_pred_bboxes_dict[image_id], 'labels': prediction.
            image_id_to_pred_classes_dict[image_id], 'masks': Preprocessor.
            inv_process_probmasks(prediction.image_id_to_process_dict_dict[
            image_id], prediction.image_id_to_pred_probmasks_dict[image_id]
            ) if pred_needs_inv_process else prediction.
            image_id_to_pred_probmasks_dict[image_id], 'scores': prediction
            .image_id_to_pred_probs_dict[image_id]}
        res = {image_id: output}
        coco_evaluator.update(res)
    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    class_to_inter_recall_array_dict = {c: np.linspace(0, 1, 101) for c in
        range(1, self._num_classes)}
    if quality == Evaluator.Evaluation.Quality.STANDARD:
        iou_threshold_index = 0
    elif quality == Evaluator.Evaluation.Quality.STRICT:
        iou_threshold_index = 5
    elif quality == Evaluator.Evaluation.Quality.STRICTEST:
        iou_threshold_index = 9
    else:
        raise ValueError
    if size == Evaluator.Evaluation.Size.ALL:
        area_index = 0
    elif size == Evaluator.Evaluation.Size.AREA_S:
        area_index = 1
    elif size == Evaluator.Evaluation.Size.AREA_M:
        area_index = 2
    elif size == Evaluator.Evaluation.Size.AREA_L:
        area_index = 3
    else:
        raise ValueError
    max_detections_index = 2
    precisions = coco_evaluator.coco_eval['segm'].eval['precision']
    scores = coco_evaluator.coco_eval['segm'].eval['scores']
    class_to_inter_precision_array_dict = {c: precisions[
        iou_threshold_index, :, c - 1, area_index, max_detections_index] for
        c in range(1, self._num_classes)}
    class_to_prob_array_dict = {c: scores[iou_threshold_index, :, c - 1,
        area_index, max_detections_index] for c in range(1, self._num_classes)}
    class_to_recall_array_dict = class_to_inter_recall_array_dict
    class_to_precision_array_dict = class_to_inter_precision_array_dict
    class_to_f1_score_array_dict = {c: (2 * class_to_recall_array_dict[c] *
        class_to_precision_array_dict[c] / (class_to_recall_array_dict[c] +
        class_to_precision_array_dict[c] + np.finfo(np.float32).eps)) for c in
        range(1, self._num_classes)}
    mean_ap = coco_evaluator.coco_eval['segm'].stats[1]
    class_to_ap_dict = {c: precisions[iou_threshold_index, :, c - 1,
        area_index, max_detections_index].mean().item() for c in range(1,
        self._num_classes)}
    evaluation = Evaluator.Evaluation(quality, size,
        class_to_inter_recall_array_dict,
        class_to_inter_precision_array_dict, class_to_recall_array_dict,
        class_to_precision_array_dict, class_to_f1_score_array_dict,
        class_to_prob_array_dict, metric_ap=Evaluator.Evaluation.
        MetricResult(mean_ap, class_to_ap_dict))
    return evaluation
