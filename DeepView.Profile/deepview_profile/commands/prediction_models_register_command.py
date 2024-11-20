def register_command(subparsers):
    parser = subparsers.add_parser('prediction-models', help=
        "Evaluate DeepView's prediction accuracy.")
    parser.add_argument('entry_point', help=
        'The entry point file in this project that contains the DeepView provider functions.'
        )
    parser.add_argument('-b', '--batch-sizes', help=
        'The starting batch sizes to build models from.', type=int, nargs=
        '+', required=True)
    parser.add_argument('-o', '--output', help=
        'The location where the evaluation output should be stored.',
        required=True)
    parser.add_argument('--log-file', help='The location of the log file.')
    parser.add_argument('--debug', action='store_true', help=
        'Log debug messages.')
    parser.set_defaults(func=main)
