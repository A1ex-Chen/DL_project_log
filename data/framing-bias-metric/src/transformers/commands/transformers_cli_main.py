def main():
    parser = ArgumentParser('Transformers CLI tool', usage=
        'transformers-cli <command> [<args>]')
    commands_parser = parser.add_subparsers(help=
        'transformers-cli command helpers')
    ConvertCommand.register_subcommand(commands_parser)
    DownloadCommand.register_subcommand(commands_parser)
    EnvironmentCommand.register_subcommand(commands_parser)
    RunCommand.register_subcommand(commands_parser)
    ServeCommand.register_subcommand(commands_parser)
    UserCommands.register_subcommand(commands_parser)
    AddNewModelCommand.register_subcommand(commands_parser)
    args = parser.parse_args()
    if not hasattr(args, 'func'):
        parser.print_help()
        exit(1)
    service = args.func(args)
    service.run()
