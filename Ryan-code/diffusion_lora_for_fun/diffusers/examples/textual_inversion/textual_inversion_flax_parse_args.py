def parse_args():
    parser = argparse.ArgumentParser(description=
        'Simple example of a training script.')
    parser.add_argument('--pretrained_model_name_or_path', type=str,
        default=None, required=True, help=
        'Path to pretrained model or model identifier from huggingface.co/models.'
        )
    parser.add_argument('--tokenizer_name', type=str, default=None, help=
        'Pretrained tokenizer name or path if not the same as model_name')
    parser.add_argument('--train_data_dir', type=str, default=None,
        required=True, help='A folder containing the training data.')
    parser.add_argument('--placeholder_token', type=str, default=None,
        required=True, help='A token to use as a placeholder for the concept.')
    parser.add_argument('--initializer_token', type=str, default=None,
        required=True, help='A token to use as initializer word.')
    parser.add_argument('--learnable_property', type=str, default='object',
        help="Choose between 'object' and 'style'")
    parser.add_argument('--repeats', type=int, default=100, help=
        'How many times to repeat the training data.')
    parser.add_argument('--output_dir', type=str, default=
        'text-inversion-model', help=
        'The output directory where the model predictions and checkpoints will be written.'
        )
    parser.add_argument('--seed', type=int, default=42, help=
        'A seed for reproducible training.')
    parser.add_argument('--resolution', type=int, default=512, help=
        'The resolution for input images, all the images in the train/validation dataset will be resized to this resolution'
        )
    parser.add_argument('--center_crop', action='store_true', help=
        'Whether to center crop images before resizing to resolution.')
    parser.add_argument('--train_batch_size', type=int, default=16, help=
        'Batch size (per device) for the training dataloader.')
    parser.add_argument('--num_train_epochs', type=int, default=100)
    parser.add_argument('--max_train_steps', type=int, default=5000, help=
        'Total number of training steps to perform.  If provided, overrides num_train_epochs.'
        )
    parser.add_argument('--save_steps', type=int, default=500, help=
        'Save learned_embeds.bin every X updates steps.')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help
        ='Initial learning rate (after the potential warmup period) to use.')
    parser.add_argument('--scale_lr', action='store_true', default=True,
        help=
        'Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.'
        )
    parser.add_argument('--lr_warmup_steps', type=int, default=500, help=
        'Number of steps for the warmup in the lr scheduler.')
    parser.add_argument('--revision', type=str, default=None, required=
        False, help=
        'Revision of pretrained model identifier from huggingface.co/models.')
    parser.add_argument('--lr_scheduler', type=str, default='constant',
        help=
        'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]'
        )
    parser.add_argument('--adam_beta1', type=float, default=0.9, help=
        'The beta1 parameter for the Adam optimizer.')
    parser.add_argument('--adam_beta2', type=float, default=0.999, help=
        'The beta2 parameter for the Adam optimizer.')
    parser.add_argument('--adam_weight_decay', type=float, default=0.01,
        help='Weight decay to use.')
    parser.add_argument('--adam_epsilon', type=float, default=1e-08, help=
        'Epsilon value for the Adam optimizer')
    parser.add_argument('--push_to_hub', action='store_true', help=
        'Whether or not to push the model to the Hub.')
    parser.add_argument('--use_auth_token', action='store_true', help=
        'Will use the token generated when running `huggingface-cli login` (necessary to use this script with private models).'
        )
    parser.add_argument('--hub_token', type=str, default=None, help=
        'The token to use to push to the Model Hub.')
    parser.add_argument('--hub_model_id', type=str, default=None, help=
        'The name of the repository to keep in sync with the local `output_dir`.'
        )
    parser.add_argument('--logging_dir', type=str, default='logs', help=
        '[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***.'
        )
    parser.add_argument('--local_rank', type=int, default=-1, help=
        'For distributed training: local_rank')
    args = parser.parse_args()
    env_local_rank = int(os.environ.get('LOCAL_RANK', -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    if args.train_data_dir is None:
        raise ValueError('You must specify a train data directory.')
    return args
