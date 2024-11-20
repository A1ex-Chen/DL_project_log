def _fetch_annotations(img_idx, image_path, batch, prediction_metadata_map,
    class_label_map):
    """Join the ground truth and prediction annotations if they exist."""
    ground_truth_annotations = _format_ground_truth_annotations_for_detection(
        img_idx, image_path, batch, class_label_map)
    prediction_annotations = _format_prediction_annotations_for_detection(
        image_path, prediction_metadata_map, class_label_map)
    annotations = [annotation for annotation in [ground_truth_annotations,
        prediction_annotations] if annotation is not None]
    return [annotations] if annotations else None
