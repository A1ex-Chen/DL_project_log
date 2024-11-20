def get_parser():
    parser = argparse.ArgumentParser(description='Uno infer script')
    parser.add_argument('--data', required=True, help=
        'data file to infer on. expect exported file from uno_baseline_keras2.py'
        )
    parser.add_argument('--model_file', required=True, help=
        'json model description file')
    parser.add_argument('--weights_file', help='model weights file')
    parser.add_argument('--partition', default='all', choices=['train',
        'val', 'all'], help='partition of test dataset')
    parser.add_argument('-n', '--n_pred', type=int, default=1, help=
        'the number of predictions to make')
    parser.add_argument('--single', default=False, help=
        'do not use drug pair representation')
    parser.add_argument('--agg_dose', default=None, choices=['AUC', 'IC50',
        'HS', 'AAC1', 'AUC1', 'DSS1'], help=
        'use dose-independent response data with the specified aggregation metric'
        )
    return parser
