def get_model(model_or_name, threads=-1, classify=False, seed=0):
    regression_models = {'xgboost': (XGBRegressor(max_depth=6, n_jobs=
        threads, random_state=seed), 'XGBRegressor'), 'lightgbm': (
        LGBMRegressor(n_jobs=threads, random_state=seed, verbose=-1),
        'LGBMRegressor'), 'randomforest': (RandomForestRegressor(
        n_estimators=100, n_jobs=threads), 'RandomForestRegressor'),
        'adaboost': (AdaBoostRegressor(), 'AdaBoostRegressor'), 'linear': (
        LinearRegression(), 'LinearRegression'), 'elasticnet': (
        ElasticNetCV(positive=True), 'ElasticNetCV'), 'lasso': (LassoCV(
        positive=True), 'LassoCV'), 'ridge': (Ridge(), 'Ridge'), 'xgb.1k':
        (XGBRegressor(max_depth=6, n_estimators=1000, n_jobs=threads,
        random_state=seed), 'XGBRegressor.1K'), 'xgb.10k': (XGBRegressor(
        max_depth=6, n_estimators=10000, n_jobs=threads, random_state=seed),
        'XGBRegressor.10K'), 'lgbm.1k': (LGBMRegressor(n_estimators=1000,
        n_jobs=threads, random_state=seed, verbose=-1), 'LGBMRegressor.1K'),
        'lgbm.10k': (LGBMRegressor(n_estimators=10000, n_jobs=threads,
        random_state=seed, verbose=-1), 'LGBMRegressor.10K'), 'rf.1k': (
        RandomForestRegressor(n_estimators=1000, n_jobs=threads),
        'RandomForestRegressor.1K'), 'rf.10k': (RandomForestRegressor(
        n_estimators=10000, n_jobs=threads), 'RandomForestRegressor.10K')}
    classification_models = {'xgboost': (XGBClassifier(max_depth=6, n_jobs=
        threads, random_state=seed), 'XGBClassifier'), 'lightgbm': (
        LGBMClassifier(n_jobs=threads, random_state=seed, verbose=-1),
        'LGBMClassifier'), 'randomforest': (RandomForestClassifier(
        n_estimators=100, n_jobs=threads), 'RandomForestClassifier'),
        'adaboost': (AdaBoostClassifier(), 'AdaBoostClassifier'),
        'logistic': (LogisticRegression(), 'LogisticRegression'),
        'gaussian': (GaussianProcessClassifier(),
        'GaussianProcessClassifier'), 'knn': (KNeighborsClassifier(),
        'KNeighborsClassifier'), 'bayes': (GaussianNB(), 'GaussianNB'),
        'svm': (SVC(), 'SVC'), 'xgb.1k': (XGBClassifier(max_depth=6,
        n_estimators=1000, n_jobs=threads, random_state=seed),
        'XGBClassifier.1K'), 'xgb.10k': (XGBClassifier(max_depth=6,
        n_estimators=10000, n_jobs=threads, random_state=seed),
        'XGBClassifier.10K'), 'lgbm.1k': (LGBMClassifier(n_estimators=1000,
        n_jobs=threads, random_state=seed, verbose=-1), 'LGBMClassifier.1K'
        ), 'lgbm.10k': (LGBMClassifier(n_estimators=1000, n_jobs=threads,
        random_state=seed, verbose=-1), 'LGBMClassifier.10K'), 'rf.1k': (
        RandomForestClassifier(n_estimators=1000, n_jobs=threads),
        'RandomForestClassifier.1K'), 'rf.10k': (RandomForestClassifier(
        n_estimators=10000, n_jobs=threads), 'RandomForestClassifier.10K')}
    if isinstance(model_or_name, str):
        if classify:
            model_and_name = classification_models.get(model_or_name.lower())
        else:
            model_and_name = regression_models.get(model_or_name.lower())
        if not model_and_name:
            raise Exception("unrecognized model: '{}'".format(model_or_name))
        else:
            model, name = model_and_name
    else:
        model = model_or_name
        name = re.search('\\w+', str(model)).group(0)
    return model, name
