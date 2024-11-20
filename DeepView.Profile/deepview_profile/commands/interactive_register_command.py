def register_command(subparsers):
    parser = subparsers.add_parser('interactive', help=
        'Start a new DeepView interactive profiling session.')
    parser.add_argument('--host', default='', help=
        'The host address to bind to.')
    parser.add_argument('--port', default=60120, type=int, help=
        'The port to listen on.')
    parser.add_argument('--hints-file', help=
        'Path to the performance hints configuration YAML file.')
    parser.add_argument('--measure-for', help=
        'Number of iterations to measure when determining throughput.')
    parser.add_argument('--warm-up', help=
        'Number of warm up iterations when determining throughput.')
    parser.add_argument('--log-file', help='The location of the log file.')
    parser.add_argument('--debug', action='store_true', help=
        'Log debug messages.')
    parser.set_defaults(func=main)
