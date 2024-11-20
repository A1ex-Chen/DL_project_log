@staticmethod
def _write_coco_annotation(path_to_annotation_json: str,
    unfolded_numeric_image_ids: List[int], unfolded_gt_bboxes: np.ndarray,
    unfolded_gt_classes: np.ndarray, unfolded_gt_difficulties: np.ndarray,
    num_classes: int):
    images = []
    categories = []
    annotations = []
    for numeric_image_id in set(unfolded_numeric_image_ids):
        images.append({'id': numeric_image_id})
    for i, (numeric_image_id, bbox, cls, diff) in enumerate(zip(
        unfolded_numeric_image_ids, unfolded_gt_bboxes.tolist(),
        unfolded_gt_classes.tolist(), unfolded_gt_difficulties.tolist())):
        annotations.append({'id': i + 1, 'image_id': numeric_image_id,
            'bbox': [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
            ], 'area': (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
            'category_id': cls, 'iscrowd': diff})
    for cls in range(1, num_classes):
        categories.append({'id': cls})
    with open(path_to_annotation_json, 'w') as f:
        json.dump({'images': images, 'annotations': annotations,
            'categories': categories}, f)
