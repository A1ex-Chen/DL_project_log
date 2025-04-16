def _add_bool_argument(parser, name=None, default=False, required=False,
    help=None):
    if not isinstance(default, bool):
        raise ValueError()
    feature_parser = parser.add_mutually_exclusive_group(required=required)
    feature_parser.add_argument('--' + name, dest=name, action='store_true',
        help=help, default=default)
    feature_parser.add_argument('--no' + name, dest=name, action='store_false')
    feature_parser.set_defaults(name=default)
