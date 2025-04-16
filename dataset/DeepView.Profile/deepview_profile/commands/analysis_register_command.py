def register_command(subparsers):
    parser = subparsers.add_parser('analysis', help=
        'Generate usage report for various analysis.')
    parser.add_argument('entry_point', help=
        'The entry point file in this project that contains the DeepView provider functions.'
        )
    parser.add_argument('--all', action='store_true', help=
        'The complete analysis of all methods')
    parser.add_argument('-breakdown', '--measure-breakdown', action=
        'store_true', help='Adds breakdown data to results')
    parser.add_argument('-throughput', '--measure-throughput', action=
        'store_true', help='Adds throughput data to results')
    parser.add_argument('-predict', '--habitat-predict', action=
        'store_true', help='Adds habitat data prediction to results')
    parser.add_argument('-utilization', '--measure-utilization', action=
        'store_true', help='Adds utilization data to results')
    parser.add_argument('-energy', '--energy-compute', action='store_true',
        help='Adds energy use to results')
    parser.add_argument('--include-ddp', action='store_true', help=
        'Adds ddp analysis to results')
    parser.add_argument('-o', '--output', help=
        'The location where the complete report should be stored', required
        =True)
    parser.add_argument('--log-file', help='The location of the log file')
    parser.add_argument('--exclude-source', action='store_true', help=
        'Allows not adding encodedFiles section')
    parser.add_argument('--debug', action='store_true', help=
        'Log debug messages.')
    parser.set_defaults(func=main)
