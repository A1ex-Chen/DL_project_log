def iou_distance(atracks: list, btracks: list) ->np.ndarray:
    """
    Compute cost based on Intersection over Union (IoU) between tracks.

    Args:
        atracks (list[STrack] | list[np.ndarray]): List of tracks 'a' or bounding boxes.
        btracks (list[STrack] | list[np.ndarray]): List of tracks 'b' or bounding boxes.

    Returns:
        (np.ndarray): Cost matrix computed based on IoU.
    """
    if atracks and isinstance(atracks[0], np.ndarray
        ) or btracks and isinstance(btracks[0], np.ndarray):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [(track.xywha if track.angle is not None else track.xyxy) for
            track in atracks]
        btlbrs = [(track.xywha if track.angle is not None else track.xyxy) for
            track in btracks]
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)
    if len(atlbrs) and len(btlbrs):
        if len(atlbrs[0]) == 5 and len(btlbrs[0]) == 5:
            ious = batch_probiou(np.ascontiguousarray(atlbrs, dtype=np.
                float32), np.ascontiguousarray(btlbrs, dtype=np.float32)
                ).numpy()
        else:
            ious = bbox_ioa(np.ascontiguousarray(atlbrs, dtype=np.float32),
                np.ascontiguousarray(btlbrs, dtype=np.float32), iou=True)
    return 1 - ious
