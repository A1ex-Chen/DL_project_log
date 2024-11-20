@staticmethod
def register_subcommand(parser: ArgumentParser):
    add_new_model_parser = parser.add_parser('add-new-model')
    add_new_model_parser.add_argument('--testing', action='store_true',
        help='If in testing mode.')
    add_new_model_parser.add_argument('--testing_file', type=str, help=
        'Configuration file on which to run.')
    add_new_model_parser.add_argument('--path', type=str, help=
        'Path to cookiecutter. Should only be used for testing purposes.')
    add_new_model_parser.set_defaults(func=add_new_model_command_factory)
