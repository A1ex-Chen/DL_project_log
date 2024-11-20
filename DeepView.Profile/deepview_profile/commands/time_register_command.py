def register_command(subparsers):
    parser = subparsers.add_parser('time', help=
        'Generate an iteration run time breakdown report.')
    parser.add_argument('entry_point', help=
        'The entry point file in this project that contains the DeepView provider functions.'
        )
    parser.add_argument('-o', '--output', help=
        'The location where the iteration run time breakdown report should be stored.'
        , required=True)
    parser.add_argument('--log-file', help='The location of the log file.')
    parser.add_argument('--debug', action='store_true', help=
        'Log debug messages.')
    parser.set_defaults(func=main)
