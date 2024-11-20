def add_parser_arguments(parser):
    parser.add_argument('--checkpoint-path', metavar='<path>', help=
        'checkpoint filename')
    parser.add_argument('--weight-path', metavar='<path>', help=
        'name of file in which to store weights')
