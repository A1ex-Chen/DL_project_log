def on_predict_postprocess_end(predictor: object, persist: bool=False) ->None:
    """
    Postprocess detected boxes and update with object tracking.

    Args:
        predictor (object): The predictor object containing the predictions.
        persist (bool, optional): Whether to persist the trackers if they already exist. Defaults to False.
    """
    path, im0s = predictor.batch[:2]
    is_obb = predictor.args.task == 'obb'
    is_stream = predictor.dataset.mode == 'stream'
    for i in range(len(im0s)):
        tracker = predictor.trackers[i if is_stream else 0]
        vid_path = predictor.save_dir / Path(path[i]).name
        if not persist and predictor.vid_path[i if is_stream else 0
            ] != vid_path:
            tracker.reset()
            predictor.vid_path[i if is_stream else 0] = vid_path
        det = (predictor.results[i].obb if is_obb else predictor.results[i]
            .boxes).cpu().numpy()
        if len(det) == 0:
            continue
        tracks = tracker.update(det, im0s[i])
        if len(tracks) == 0:
            continue
        idx = tracks[:, -1].astype(int)
        predictor.results[i] = predictor.results[i][idx]
        update_args = {('obb' if is_obb else 'boxes'): torch.as_tensor(
            tracks[:, :-1])}
        predictor.results[i].update(**update_args)
