@staticmethod
def _write_coco_results(path_to_results_json: str,
    unfolded_numeric_image_ids: List[int], unfolded_pred_bboxes: np.ndarray,
    unfolded_pred_classes: np.ndarray, unfolded_pred_probs: np.ndarray):
    results = []
    for numeric_image_id, bbox, cls, prob in zip(unfolded_numeric_image_ids,
        unfolded_pred_bboxes.tolist(), unfolded_pred_classes.tolist(),
        unfolded_pred_probs.tolist()):
        results.append({'image_id': numeric_image_id, 'category_id': cls,
            'bbox': [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
            ], 'score': prob})
    with open(path_to_results_json, 'w') as f:
        json.dump(results, f)
