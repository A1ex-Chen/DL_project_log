@staticmethod
def register_subcommand(parser: ArgumentParser):
    """
        Register this command to argparse so it's available for the transformer-cli

        Args:
            parser: Root parser to register command-specific arguments
        """
    serve_parser = parser.add_parser('serve', help=
        'CLI tool to run inference requests through REST and GraphQL endpoints.'
        )
    serve_parser.add_argument('--task', type=str, choices=SUPPORTED_TASKS.
        keys(), help='The task to run the pipeline on')
    serve_parser.add_argument('--host', type=str, default='localhost', help
        ='Interface the server will listen on.')
    serve_parser.add_argument('--port', type=int, default=8888, help=
        'Port the serving will listen to.')
    serve_parser.add_argument('--workers', type=int, default=1, help=
        'Number of http workers')
    serve_parser.add_argument('--model', type=str, help=
        "Model's name or path to stored model.")
    serve_parser.add_argument('--config', type=str, help=
        "Model's config name or path to stored model.")
    serve_parser.add_argument('--tokenizer', type=str, help=
        'Tokenizer name to use.')
    serve_parser.add_argument('--device', type=int, default=-1, help=
        'Indicate the device to run onto, -1 indicates CPU, >= 0 indicates GPU (default: -1)'
        )
    serve_parser.set_defaults(func=serve_command_factory)
