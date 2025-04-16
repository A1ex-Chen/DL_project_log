@staticmethod
def register_subcommand(parser: ArgumentParser):
    """
        Register this command to argparse so it's available for the transformer-cli

        Args:
            parser: Root parser to register command-specific arguments
        """
    train_parser = parser.add_parser('train', help=
        'CLI tool to train a model on a task.')
    train_parser.add_argument('--train_data', type=str, required=True, help
        =
        'path to train (and optionally evaluation) dataset as a csv with tab separated labels and sentences.'
        )
    train_parser.add_argument('--column_label', type=int, default=0, help=
        'Column of the dataset csv file with example labels.')
    train_parser.add_argument('--column_text', type=int, default=1, help=
        'Column of the dataset csv file with example texts.')
    train_parser.add_argument('--column_id', type=int, default=2, help=
        'Column of the dataset csv file with example ids.')
    train_parser.add_argument('--skip_first_row', action='store_true', help
        ='Skip the first row of the csv file (headers).')
    train_parser.add_argument('--validation_data', type=str, default='',
        help='path to validation dataset.')
    train_parser.add_argument('--validation_split', type=float, default=0.1,
        help=
        'if validation dataset is not provided, fraction of train dataset to use as validation dataset.'
        )
    train_parser.add_argument('--output', type=str, default='./', help=
        'path to saved the trained model.')
    train_parser.add_argument('--task', type=str, default=
        'text_classification', help='Task to train the model on.')
    train_parser.add_argument('--model', type=str, default=
        'bert-base-uncased', help="Model's name or path to stored model.")
    train_parser.add_argument('--train_batch_size', type=int, default=32,
        help='Batch size for training.')
    train_parser.add_argument('--valid_batch_size', type=int, default=64,
        help='Batch size for validation.')
    train_parser.add_argument('--learning_rate', type=float, default=3e-05,
        help='Learning rate.')
    train_parser.add_argument('--adam_epsilon', type=float, default=1e-08,
        help='Epsilon for Adam optimizer.')
    train_parser.set_defaults(func=train_command_factory)
