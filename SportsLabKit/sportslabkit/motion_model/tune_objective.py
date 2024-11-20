def objective(trial: optuna.Trial):
    params = {}
    for param_name, search_space in hparam_search_space.items():
        if search_space['type'] == 'categorical':
            params[param_name] = trial.suggest_categorical(param_name,
                search_space['values'])
        elif search_space['type'] == 'float':
            params[param_name] = trial.suggest_float(param_name,
                search_space['low'], search_space['high'])
        elif search_space['type'] == 'logfloat':
            params[param_name] = trial.suggest_float(param_name,
                search_space['low'], search_space['high'], log=True)
        elif search_space['type'] == 'int':
            params[param_name] = trial.suggest_int(param_name, search_space
                ['low'], search_space['high'])
    motion_model = motion_model_class(model_config=params)
    tracklet = Tracklet()
    tracklet.register_observation_types(['box', 'score'])
    ious = []
    for det, gt in zip(detections, ground_truth_positions):
        obs = {'box': det.box, 'score': det.score}
        tracklet.update_observations(obs)
        prediction = motion_model(tracklet)
        iou = iou_score(convert_to_x1y1x2y2(prediction),
            convert_to_x1y1x2y2(gt))
        ious.append(iou)
    avg_iou = np.mean(ious)
    return 1 - avg_iou
