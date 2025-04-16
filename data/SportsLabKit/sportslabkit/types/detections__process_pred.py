def _process_pred(self, pred: (dict | list | Detection)) ->np.ndarray:
    if isinstance(pred, dict):
        if len(pred.keys()) != 6:
            raise ValueError(
                'The prediction dictionary should contain exactly 6 items')
        return np.stack([pred['bbox_left'], pred['bbox_top'], pred[
            'bbox_width'], pred['bbox_height'], pred['conf'], pred['class']
            ], axis=0)
    elif isinstance(pred, list):
        if len(pred) != 6:
            raise ValueError(
                'The prediction list should contain exactly 6 items')
        return np.array(pred)
    elif isinstance(pred, Detection):
        return np.array([pred.box[0], pred.box[1], pred.box[2], pred.box[3],
            pred.score, pred.class_id])
    elif isinstance(pred, np.ndarray):
        if pred.shape != (6,):
            raise ValueError(
                f'pred should have the shape (6, ), but got {pred.shape}')
        return pred
    else:
        raise TypeError(f'Unsupported prediction type: {type(pred)}')
