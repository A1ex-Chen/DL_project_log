@staticmethod
def register_subcommand(parser: ArgumentParser):
    download_parser = parser.add_parser('download')
    download_parser.add_argument('--cache-dir', type=str, default=None,
        help='Path to location to store the models')
    download_parser.add_argument('--force', action='store_true', help=
        'Force the model to be download even if already in cache-dir')
    download_parser.add_argument('model', type=str, help=
        'Name of the model to download')
    download_parser.set_defaults(func=download_command_factory)
