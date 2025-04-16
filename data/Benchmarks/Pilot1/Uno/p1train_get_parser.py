def get_parser(description=
    'Run machine learning training algorithms implemented in scikit-learn'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-b', '--bins', type=int, default=BINS, help=
        'number of evenly distributed bins to make when classification mode is turned on'
        )
    parser.add_argument('-c', '--classify', action='store_true', help=
        'convert the regression problem into classification based on category cutoffs'
        )
    parser.add_argument('-d', '--data', help='data file to train on')
    parser.add_argument('-g', '--groupcols', nargs='+', help=
        'names of columns to be used in cross validation partitioning')
    parser.add_argument('-m', '--models', nargs='+', default=MODELS, help=
        'list of regression models: XGBoost, XGB.1K, XGB.10K, RandomForest, RF.1K, RF.10K, AdaBoost, Linear, ElasticNet, Lasso, Ridge; or list of classification models: XGBoost, XGB.1K, XGB.10K, RandomForest, RF.1K, RF.10K, AdaBoost, Logistic, Gaussian, Bayes, KNN, SVM'
        )
    parser.add_argument('-o', '--out_dir', default=OUT_DIR, help=
        'output directory')
    parser.add_argument('-p', '--prefix', help='output prefix')
    parser.add_argument('-t', '--threads', type=int, default=THREADS, help=
        'number of threads per machine learning training job; -1 for using all threads'
        )
    parser.add_argument('-y', '--ycol', default='0', help=
        '0-based index or name of the column to be predicted')
    parser.add_argument('--cutoffs', nargs='+', type=float, default=CUTOFFS,
        help='list of cutoffs delineating prediction target categories')
    parser.add_argument('--cv', type=int, default=CV, help=
        'cross validation folds')
    parser.add_argument('--feature_subsample', type=int, default=
        FEATURE_SUBSAMPLE, help=
        'number of features to randomly sample from each category, 0 means using all features'
        )
    parser.add_argument('-C', '--ignore_categoricals', action='store_true',
        help='ignore categorical feature columns')
    parser.add_argument('--balanced', action='store_true', help=
        'balanced class weights')
    parser.add_argument('--csv', action='store_true', help=
        'comma separated file')
    parser.add_argument('--seed', type=int, default=SEED, help=
        'specify random seed')
    return parser
