def main():
    parser = ArgumentParser('Diffusers CLI tool', usage=
        'diffusers-cli <command> [<args>]')
    commands_parser = parser.add_subparsers(help=
        'diffusers-cli command helpers')
    EnvironmentCommand.register_subcommand(commands_parser)
    args = parser.parse_args()
    if not hasattr(args, 'func'):
        parser.print_help()
        exit(1)
    service = args.func(args)
    service.run()
