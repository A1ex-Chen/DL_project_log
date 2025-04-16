def get_common_parser(parser):
    """Parse command-line arguments. Ignore if not present.

    Parameters
    ----------
    parser : ArgumentParser object
        Parser for command-line options
    """
    parser.add_argument('--config_file', dest='config_file', default=
        argparse.SUPPRESS, help='specify model configuration file')
    parser.add_argument('--train_bool', dest='train_bool', type=str2bool,
        default=True, help='train model')
    parser.add_argument('--eval_bool', dest='eval_bool', type=str2bool,
        default=argparse.SUPPRESS, help='evaluate model (use it for inference)'
        )
    parser.add_argument('--timeout', dest='timeout', type=int, action=
        'store', default=argparse.SUPPRESS, help=
        'seconds allowed to train model (default: no timeout)')
    parser.add_argument('--home_dir', dest='home_dir', default=argparse.
        SUPPRESS, type=str, help='set home directory')
    parser.add_argument('--train_data', action='store', default=argparse.
        SUPPRESS, help='training data filename')
    parser.add_argument('--val_data', action='store', default=argparse.
        SUPPRESS, help='validation data filename')
    parser.add_argument('--test_data', action='store', default=argparse.
        SUPPRESS, help='testing data filename')
    parser.add_argument('--output_dir', dest='output_dir', default=argparse
        .SUPPRESS, type=str, help='output directory')
    parser.add_argument('--data_url', dest='data_url', default=argparse.
        SUPPRESS, type=str, help='set data source url')
    parser.add_argument('--experiment_id', default='EXP000', type=str, help
        ='set the experiment unique identifier')
    parser.add_argument('--run_id', default='RUN000', type=str, help=
        'set the run unique identifier')
    parser.add_argument('--conv', nargs='+', type=int, default=argparse.
        SUPPRESS, help=
        'integer array describing convolution layers: conv1_filters, conv1_filter_len, conv1_stride, conv2_filters, conv2_filter_len, conv2_stride ...'
        )
    parser.add_argument('--locally_connected', type=str2bool, default=
        argparse.SUPPRESS, help=
        'use locally connected layers instead of convolution layers')
    parser.add_argument('-a', '--activation', default=argparse.SUPPRESS,
        help=
        'keras activation function to use in inner layers: relu, tanh, sigmoid...'
        )
    parser.add_argument('--out_activation', default=argparse.SUPPRESS, help
        ='keras activation function to use in out layer: softmax, linear, ...')
    parser.add_argument('--lstm_size', nargs='+', type=int, default=
        argparse.SUPPRESS, help=
        'integer array describing size of LSTM internal state per layer')
    parser.add_argument('--recurrent_dropout', action='store', default=
        argparse.SUPPRESS, type=float, help='ratio of recurrent dropout')
    parser.add_argument('--dropout', type=float, default=argparse.SUPPRESS,
        help='ratio of dropout used in fully connected layers')
    parser.add_argument('--pool', type=int, default=argparse.SUPPRESS, help
        ='pooling layer length')
    parser.add_argument('--batch_normalization', type=str2bool, default=
        argparse.SUPPRESS, help='use batch normalization')
    parser.add_argument('--loss', default=argparse.SUPPRESS, help=
        'keras loss function to use: mse, ...')
    parser.add_argument('--optimizer', default=argparse.SUPPRESS, help=
        'keras optimizer to use: sgd, rmsprop, ...')
    parser.add_argument('--metrics', default=argparse.SUPPRESS, help=
        'metrics to evaluate performance: accuracy, ...')
    parser.add_argument('--scaling', default=argparse.SUPPRESS, choices=[
        'minabs', 'minmax', 'std', 'none'], help=
        "type of feature scaling; 'minabs': to [-1,1]; 'minmax': to [0,1], 'std': standard unit normalization; 'none': no normalization"
        )
    parser.add_argument('--shuffle', type=str2bool, default=False, help=
        'randomly shuffle data set (produces different training and testing partitions each run depending on the seed)'
        )
    parser.add_argument('--feature_subsample', type=int, default=argparse.
        SUPPRESS, help=
        'number of features to randomly sample from each category (cellline expression, drug descriptors, etc), 0 means using all features'
        )
    parser.add_argument('--learning_rate', default=argparse.SUPPRESS, type=
        float, help='overrides the learning rate for training')
    parser.add_argument('--early_stop', type=str2bool, default=argparse.
        SUPPRESS, help=
        'activates keras callback for early stopping of training in function of the monitored variable specified'
        )
    parser.add_argument('--momentum', default=argparse.SUPPRESS, type=float,
        help='overrides the momentum to use in the SGD optimizer when training'
        )
    parser.add_argument('--initialization', default=argparse.SUPPRESS,
        choices=['constant', 'uniform', 'normal', 'glorot_uniform',
        'glorot_normal', 'lecun_uniform', 'he_normal'], help=
        "type of weight initialization; 'constant': to 0; 'uniform': to [-0.05,0.05], 'normal': mean 0, stddev 0.05; 'glorot_uniform': [-lim,lim] with lim = sqrt(6/(fan_in+fan_out)); 'lecun_uniform' : [-lim,lim] with lim = sqrt(3/fan_in); 'he_normal' : mean 0, stddev sqrt(2/fan_in)"
        )
    parser.add_argument('--val_split', type=float, default=argparse.
        SUPPRESS, help='fraction of data to use in validation')
    parser.add_argument('--train_steps', type=int, default=argparse.
        SUPPRESS, help=
        'overrides the number of training batches per epoch if set to nonzero')
    parser.add_argument('--val_steps', type=int, default=argparse.SUPPRESS,
        help=
        'overrides the number of validation batches per epoch if set to nonzero'
        )
    parser.add_argument('--test_steps', type=int, default=argparse.SUPPRESS,
        help='overrides the number of test batches per epoch if set to nonzero'
        )
    parser.add_argument('--train_samples', action='store', default=argparse
        .SUPPRESS, type=int, help=
        'overrides the number of training samples if set to nonzero')
    parser.add_argument('--val_samples', action='store', default=argparse.
        SUPPRESS, type=int, help=
        'overrides the number of validation samples if set to nonzero')
    parser.add_argument('--gpus', nargs='+', type=int, default=argparse.
        SUPPRESS, help='set IDs of GPUs to use')
    parser.add_argument('-p', '--profiling', type=str2bool, default='false',
        help='Turn profiling on or off')
    parser.add_argument('--clr_flag', default=argparse.SUPPRESS, type=
        str2bool, help='CLR flag (boolean)')
    parser.add_argument('--clr_mode', default=argparse.SUPPRESS, type=str,
        choices=['trng1', 'trng2', 'exp'], help='CLR mode (default: trng1)')
    parser.add_argument('--clr_base_lr', type=float, default=argparse.
        SUPPRESS, help='Base lr for cycle lr.')
    parser.add_argument('--clr_max_lr', type=float, default=argparse.
        SUPPRESS, help='Max lr for cycle lr.')
    parser.add_argument('--clr_gamma', type=float, default=argparse.
        SUPPRESS, help='Gamma parameter for learning cycle LR.')
    return parser
