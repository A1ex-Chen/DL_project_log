def evaluate_by_pycocotools(self) ->PyCOCOToolsResult:
    with tempfile.TemporaryDirectory() as path_to_temp_dir:
        path_to_annotation_json = os.path.join(path_to_temp_dir,
            'annotation.json')
        path_to_results_json = os.path.join(path_to_temp_dir, 'results.json')
        image_id_to_numeric_image_id_dict = {image_id: (i + 1) for i,
            image_id in enumerate(self.image_ids)}
        unfolded_pred_numeric_image_ids = [image_id_to_numeric_image_id_dict
            [image_id] for image_id in self.unfolded_pred_image_ids]
        unfolded_gt_numeric_image_ids = [image_id_to_numeric_image_id_dict[
            image_id] for image_id in self.unfolded_gt_image_ids]
        self._write_coco_annotation(path_to_annotation_json,
            unfolded_gt_numeric_image_ids, self.unfolded_gt_bboxes, self.
            unfolded_gt_classes, self.unfolded_gt_difficulties, self.
            num_classes)
        self._write_coco_results(path_to_results_json,
            unfolded_pred_numeric_image_ids, self.unfolded_pred_bboxes,
            self.unfolded_pred_classes, self.unfolded_pred_probs)
        cocoGt = COCO(path_to_annotation_json)
        cocoDt = cocoGt.loadRes(path_to_results_json)
        annType = 'bbox'
        cocoEval = COCOeval(cocoGt, cocoDt, annType)
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        mean_mean_ap = cocoEval.stats[0].item()
        mean_standard_ap = cocoEval.stats[1].item()
        mean_strict_ap = cocoEval.stats[2].item()
    return AveragePrecision.PyCOCOToolsResult(mean_mean_ap,
        mean_standard_ap, mean_strict_ap)
