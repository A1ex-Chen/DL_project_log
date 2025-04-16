def convert_to_detection(pred):
    """Convert an output to a single detection object.

    Handles the following input types:
    - dict with keys: bbox_left, bbox_top, bbox_width, bbox_height, conf, class
    - list or tuple with 6 items: bbox_left, bbox_top, bbox_width, bbox_height, conf, class
    - Detection object

    Args:
        pred: prediction object to convert

    Returns:
        Detection object
    """
    if isinstance(pred, dict):
        if len(pred.keys()) != 6:
            raise ValueError(
                'The prediction dictionary should contain exactly 6 items')
        return Detection(box=np.array([pred['bbox_left'], pred['bbox_top'],
            pred['bbox_width'], pred['bbox_height']]), score=pred['conf'],
            class_id=pred['class'])
    elif isinstance(pred, list) or isinstance(pred, tuple) or isinstance(pred,
        np.ndarray):
        if len(pred) != 6:
            raise ValueError(
                'The prediction list should contain exactly 6 items')
        return Detection(box=np.array(pred[:4]), score=pred[4], class_id=
            pred[5])
    elif isinstance(pred, Detection):
        return pred
    else:
        raise TypeError(f'Unsupported prediction type: {type(pred)}')
