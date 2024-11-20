def _create_prediction_metadata_map(model_predictions):
    """Create metadata map for model predictions by groupings them based on image ID."""
    pred_metadata_map = {}
    for prediction in model_predictions:
        pred_metadata_map.setdefault(prediction['image_id'], [])
        pred_metadata_map[prediction['image_id']].append(prediction)
    return pred_metadata_map
