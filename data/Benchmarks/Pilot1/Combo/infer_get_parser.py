def get_parser(description=None):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-s', '--sample_set', default='NCIPDM', help=
        'cell sample set: NCI60, NCIPDM, GDSC, RTS, ...')
    parser.add_argument('-d', '--drug_set', default='ALMANAC', help=
        'drug set: ALMANAC, GDSC, NCI_IOA_AOA, RTS, ...')
    parser.add_argument('-z', '--batch_size', type=int, default=100000,
        help='batch size')
    parser.add_argument('--step', type=int, default=10000, help=
        'number of rows to inter in each step')
    parser.add_argument('-m', '--model_file', default='saved.model.h5',
        help='trained model file')
    parser.add_argument('-n', '--n_pred', type=int, default=1, help=
        'the number of predictions to make for each sample-drug combination for uncertainty quantification'
        )
    parser.add_argument('-w', '--weights_file', default='saved.weights.h5',
        help=
        'trained weights file (loading model file alone sometimes does not work in keras)'
        )
    parser.add_argument('--ns', type=int, default=0, help=
        'the first n entries of cell samples to subsample')
    parser.add_argument('--nd', type=int, default=0, help=
        'the first n entries of drugs to subsample')
    parser.add_argument('--si', type=int, default=0, help=
        'the index of the first cell sample to subsample')
    parser.add_argument('--use_landmark_genes', action='store_true', help=
        'use the 978 landmark genes from LINCS (L1000) as expression features')
    parser.add_argument('--preprocess_rnaseq', choices=['source_scale',
        'combat', 'none'], help=
        'preprocessing method for RNAseq data; none for global normalization')
    return parser
