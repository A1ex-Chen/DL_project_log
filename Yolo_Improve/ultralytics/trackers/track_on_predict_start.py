def on_predict_start(predictor: object, persist: bool=False) ->None:
    """
    Initialize trackers for object tracking during prediction.

    Args:
        predictor (object): The predictor object to initialize trackers for.
        persist (bool, optional): Whether to persist the trackers if they already exist. Defaults to False.

    Raises:
        AssertionError: If the tracker_type is not 'bytetrack' or 'botsort'.
    """
    if hasattr(predictor, 'trackers') and persist:
        return
    tracker = check_yaml(predictor.args.tracker)
    cfg = IterableSimpleNamespace(**yaml_load(tracker))
    if cfg.tracker_type not in {'bytetrack', 'botsort'}:
        raise AssertionError(
            f"Only 'bytetrack' and 'botsort' are supported for now, but got '{cfg.tracker_type}'"
            )
    trackers = []
    for _ in range(predictor.dataset.bs):
        tracker = TRACKER_MAP[cfg.tracker_type](args=cfg, frame_rate=30)
        trackers.append(tracker)
        if predictor.dataset.mode != 'stream':
            break
    predictor.trackers = trackers
    predictor.vid_path = [None] * predictor.dataset.bs
