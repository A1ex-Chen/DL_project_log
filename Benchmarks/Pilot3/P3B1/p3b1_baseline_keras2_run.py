def run(gParameters):
    fpath = fetch_data(gParameters)
    kerasDefaults = candle.keras_default_config()
    shared_nnet_spec = gParameters['shared_nnet_spec']
    individual_nnet_spec = gParameters['ind_nnet_spec']
    features = []
    feat = gParameters['feature_names'].split(':')
    for f in feat:
        features.append(f)
    n_feat = len(feat)
    print('Feature names:')
    for i in range(n_feat):
        print(features[i])
    truth_array = [[] for _ in range(n_feat)]
    pred_array = [[] for _ in range(n_feat)]
    avg_loss = 0.0
    verbose = True
    for fold in range(gParameters['n_fold']):
        X_train, Y_train, X_test, Y_test = bmk.build_data(len(
            individual_nnet_spec), fold, fpath)
        input_dim = len(X_train[0][0])
        models, labels_train, labels_test = build_model(gParameters,
            kerasDefaults, shared_nnet_spec, individual_nnet_spec,
            input_dim, Y_train, Y_test, verbose)
        models = train_model(gParameters, models, X_train, labels_train,
            X_test, labels_test, fold, verbose)
        ret = evaluate_model(X_test, Y_test, labels_test, models)
        for i in range(n_feat):
            truth_array[i].extend(ret[i][0])
            pred_array[i].extend(ret[i][1])
        avg_loss += ret[-1]
    avg_loss /= float(gParameters['n_fold'])
    for task in range(n_feat):
        print('Task', task + 1, ':', features[task], '- Macro F1 score',
            f1_score(truth_array[task], pred_array[task], average='macro'))
        print('Task', task + 1, ':', features[task], '- Micro F1 score',
            f1_score(truth_array[task], pred_array[task], average='micro'))
    return avg_loss
