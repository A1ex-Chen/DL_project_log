@staticmethod
def register_subcommand(parser: ArgumentParser):
    download_parser = parser.add_parser('env')
    download_parser.set_defaults(func=info_command_factory)
