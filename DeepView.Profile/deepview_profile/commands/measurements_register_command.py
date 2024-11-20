def register_command(subparsers):
    parser = subparsers.add_parser('measure-batches', help=
        'Make throughput and memory measurements for given batch sizes.')
    parser.add_argument('entry_point', help=
        'The entry point file in this project that contains the DeepView provider functions.'
        )
    parser.add_argument('-b', '--batch-sizes', help=
        'The batch sizes to consider.', type=int, nargs='+', required=True)
    parser.add_argument('-t', '--trials', help=
        'Number of trials to run when making measurements.', type=int,
        required=True, default=5)
    parser.add_argument('-o', '--output', help=
        'The location where the evaluation output should be stored.',
        required=True)
    parser.add_argument('--log-file', help='The location of the log file.')
    parser.add_argument('--debug', action='store_true', help=
        'Log debug messages.')
    parser.set_defaults(func=main)
