@staticmethod
def register_subcommand(parser: ArgumentParser):
    run_parser = parser.add_parser('run', help='Run a pipeline through the CLI'
        )
    run_parser.add_argument('--task', choices=SUPPORTED_TASKS.keys(), help=
        'Task to run')
    run_parser.add_argument('--input', type=str, help=
        'Path to the file to use for inference')
    run_parser.add_argument('--output', type=str, help=
        'Path to the file that will be used post to write results.')
    run_parser.add_argument('--model', type=str, help=
        'Name or path to the model to instantiate.')
    run_parser.add_argument('--config', type=str, help=
        "Name or path to the model's config to instantiate.")
    run_parser.add_argument('--tokenizer', type=str, help=
        'Name of the tokenizer to use. (default: same as the model name)')
    run_parser.add_argument('--column', type=str, help=
        'Name of the column to use as input. (For multi columns input as QA use column1,columns2)'
        )
    run_parser.add_argument('--format', type=str, default='infer', choices=
        PipelineDataFormat.SUPPORTED_FORMATS, help='Input format to read from')
    run_parser.add_argument('--device', type=int, default=-1, help=
        'Indicate the device to run onto, -1 indicates CPU, >= 0 indicates GPU (default: -1)'
        )
    run_parser.add_argument('--overwrite', action='store_true', help=
        'Allow overwriting the output file.')
    run_parser.set_defaults(func=run_command_factory)
