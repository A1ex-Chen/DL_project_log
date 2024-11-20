@staticmethod
def register_subcommand(parser: ArgumentParser):
    """
        Register this command to argparse so it's available for the transformer-cli

        Args:
            parser: Root parser to register command-specific arguments
        """
    train_parser = parser.add_parser('convert', help=
        'CLI tool to run convert model from original author checkpoints to Transformers PyTorch checkpoints.'
        )
    train_parser.add_argument('--model_type', type=str, required=True, help
        ="Model's type.")
    train_parser.add_argument('--tf_checkpoint', type=str, required=True,
        help='TensorFlow checkpoint path or folder.')
    train_parser.add_argument('--pytorch_dump_output', type=str, required=
        True, help='Path to the PyTorch saved model output.')
    train_parser.add_argument('--config', type=str, default='', help=
        'Configuration file path or folder.')
    train_parser.add_argument('--finetuning_task_name', type=str, default=
        None, help=
        'Optional fine-tuning task name if the TF model was a finetuned model.'
        )
    train_parser.set_defaults(func=convert_command_factory)
