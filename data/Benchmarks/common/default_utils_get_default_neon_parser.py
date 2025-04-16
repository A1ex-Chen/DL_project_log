def get_default_neon_parser(parser):
    """Parse command-line arguments that are default in neon parser (and are common to all frameworks).
    Ignore if not present.

    Parameters
    ----------
    parser : ArgumentParser object
        Parser for neon default command-line options
    """
    parser.add_argument('-v', '--verbose', type=str2bool, help=
        'increase output verbosity')
    parser.add_argument('-l', '--log', dest='logfile', default=None, help=
        'log file')
    parser.add_argument('-s', '--save_path', dest='save_path', default=
        argparse.SUPPRESS, type=str, help='file path to save model snapshots')
    parser.add_argument('--model_name', dest='model_name', type=str,
        default=argparse.SUPPRESS, help=
        'specify model name to use when building filenames for saving')
    parser.add_argument('-d', '--data_type', dest='data_type', default=
        argparse.SUPPRESS, choices=['f16', 'f32', 'f64'], help=
        'default floating point')
    parser.add_argument('--dense', nargs='+', type=int, default=argparse.
        SUPPRESS, help=
        'number of units in fully connected layers in an integer array')
    parser.add_argument('-r', '--rng_seed', dest='rng_seed', type=int,
        default=argparse.SUPPRESS, help='random number generator seed')
    parser.add_argument('-e', '--epochs', type=int, default=argparse.
        SUPPRESS, help='number of training epochs')
    parser.add_argument('-z', '--batch_size', type=int, default=argparse.
        SUPPRESS, help='batch size')
    return parser
