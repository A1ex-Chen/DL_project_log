def get_parser(description=
    'Run a trained machine learningn model in inference mode on new data'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-d', '--data', help='data file to train on')
    parser.add_argument('-m', '--model_file', help='saved trained model file')
    parser.add_argument('-k', '--keepcols', nargs='+', default=[], help=
        "columns from input data file to keep in prediction file; use 'all' to keep all original columns"
        )
    parser.add_argument('-o', '--out_dir', default=OUT_DIR, help=
        'output directory')
    parser.add_argument('-p', '--prefix', help='output prefix')
    parser.add_argument('-y', '--ycol', default=None, help=
        '0-based index or name of the column to be predicted')
    parser.add_argument('-C', '--ignore_categoricals', action='store_true',
        help='ignore categorical feature columns')
    return parser
