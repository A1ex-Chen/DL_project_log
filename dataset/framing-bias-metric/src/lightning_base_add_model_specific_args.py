@staticmethod
def add_model_specific_args(parser, root_dir):
    parser.add_argument('--model_name_or_path', default=None, type=str,
        required=True, help=
        'Path to pretrained model or model identifier from huggingface.co/models'
        )
    parser.add_argument('--config_name', default='', type=str, help=
        'Pretrained config name or path if not the same as model_name')
    parser.add_argument('--tokenizer_name', default=None, type=str, help=
        'Pretrained tokenizer name or path if not the same as model_name')
    parser.add_argument('--cache_dir', default='', type=str, help=
        'Where do you want to store the pre-trained models downloaded from huggingface.co'
        )
    parser.add_argument('--encoder_layerdrop', type=float, help=
        'Encoder layer dropout probability (Optional). Goes into model.config')
    parser.add_argument('--decoder_layerdrop', type=float, help=
        'Decoder layer dropout probability (Optional). Goes into model.config')
    parser.add_argument('--dropout', type=float, help=
        'Dropout probability (Optional). Goes into model.config')
    parser.add_argument('--attention_dropout', type=float, help=
        'Attention dropout probability (Optional). Goes into model.config')
    parser.add_argument('--learning_rate', default=5e-05, type=float, help=
        'The initial learning rate for Adam.')
    parser.add_argument('--lr_scheduler', default='linear', choices=
        arg_to_scheduler_choices, metavar=arg_to_scheduler_metavar, type=
        str, help='Learning rate scheduler')
    parser.add_argument('--weight_decay', default=0.0, type=float, help=
        'Weight decay if we apply some.')
    parser.add_argument('--adam_epsilon', default=1e-08, type=float, help=
        'Epsilon for Adam optimizer.')
    parser.add_argument('--warmup_steps', default=0, type=int, help=
        'Linear warmup over warmup_steps.')
    parser.add_argument('--num_workers', default=4, type=int, help=
        'kwarg passed to DataLoader')
    parser.add_argument('--num_train_epochs', dest='max_epochs', default=3,
        type=int)
    parser.add_argument('--train_batch_size', default=32, type=int)
    parser.add_argument('--eval_batch_size', default=32, type=int)
    parser.add_argument('--adafactor', action='store_true')
