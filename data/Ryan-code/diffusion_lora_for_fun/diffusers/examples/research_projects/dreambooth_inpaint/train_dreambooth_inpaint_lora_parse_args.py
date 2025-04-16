def parse_args():
    parser = argparse.ArgumentParser(description=
        'Simple example of a training script.')
    parser.add_argument('--pretrained_model_name_or_path', type=str,
        default=None, required=True, help=
        'Path to pretrained model or model identifier from huggingface.co/models.'
        )
    parser.add_argument('--tokenizer_name', type=str, default=None, help=
        'Pretrained tokenizer name or path if not the same as model_name')
    parser.add_argument('--instance_data_dir', type=str, default=None,
        required=True, help=
        'A folder containing the training data of instance images.')
    parser.add_argument('--class_data_dir', type=str, default=None,
        required=False, help=
        'A folder containing the training data of class images.')
    parser.add_argument('--instance_prompt', type=str, default=None, help=
        'The prompt with identifier specifying the instance')
    parser.add_argument('--class_prompt', type=str, default=None, help=
        'The prompt to specify images in the same class as provided instance images.'
        )
    parser.add_argument('--with_prior_preservation', default=False, action=
        'store_true', help='Flag to add prior preservation loss.')
    parser.add_argument('--prior_loss_weight', type=float, default=1.0,
        help='The weight of prior preservation loss.')
    parser.add_argument('--num_class_images', type=int, default=100, help=
        'Minimal class images for prior preservation loss. If not have enough images, additional images will be sampled with class_prompt.'
        )
    parser.add_argument('--output_dir', type=str, default=
        'dreambooth-inpaint-model', help=
        'The output directory where the model predictions and checkpoints will be written.'
        )
    parser.add_argument('--seed', type=int, default=None, help=
        'A seed for reproducible training.')
    parser.add_argument('--resolution', type=int, default=512, help=
        'The resolution for input images, all the images in the train/validation dataset will be resized to this resolution'
        )
    parser.add_argument('--center_crop', default=False, action='store_true',
        help=
        'Whether to center crop the input images to the resolution. If not set, the images will be randomly cropped. The images will be resized to the resolution first before cropping.'
        )
    parser.add_argument('--train_text_encoder', action='store_true', help=
        'Whether to train the text encoder')
    parser.add_argument('--train_batch_size', type=int, default=4, help=
        'Batch size (per device) for the training dataloader.')
    parser.add_argument('--sample_batch_size', type=int, default=4, help=
        'Batch size (per device) for sampling images.')
    parser.add_argument('--num_train_epochs', type=int, default=1)
    parser.add_argument('--max_train_steps', type=int, default=None, help=
        'Total number of training steps to perform.  If provided, overrides num_train_epochs.'
        )
    parser.add_argument('--gradient_accumulation_steps', type=int, default=
        1, help=
        'Number of updates steps to accumulate before performing a backward/update pass.'
        )
    parser.add_argument('--gradient_checkpointing', action='store_true',
        help=
        'Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.'
        )
    parser.add_argument('--learning_rate', type=float, default=5e-06, help=
        'Initial learning rate (after the potential warmup period) to use.')
    parser.add_argument('--scale_lr', action='store_true', default=False,
        help=
        'Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.'
        )
    parser.add_argument('--lr_scheduler', type=str, default='constant',
        help=
        'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]'
        )
    parser.add_argument('--lr_warmup_steps', type=int, default=500, help=
        'Number of steps for the warmup in the lr scheduler.')
    parser.add_argument('--use_8bit_adam', action='store_true', help=
        'Whether or not to use 8-bit Adam from bitsandbytes.')
    parser.add_argument('--adam_beta1', type=float, default=0.9, help=
        'The beta1 parameter for the Adam optimizer.')
    parser.add_argument('--adam_beta2', type=float, default=0.999, help=
        'The beta2 parameter for the Adam optimizer.')
    parser.add_argument('--adam_weight_decay', type=float, default=0.01,
        help='Weight decay to use.')
    parser.add_argument('--adam_epsilon', type=float, default=1e-08, help=
        'Epsilon value for the Adam optimizer')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, help=
        'Max gradient norm.')
    parser.add_argument('--push_to_hub', action='store_true', help=
        'Whether or not to push the model to the Hub.')
    parser.add_argument('--hub_token', type=str, default=None, help=
        'The token to use to push to the Model Hub.')
    parser.add_argument('--hub_model_id', type=str, default=None, help=
        'The name of the repository to keep in sync with the local `output_dir`.'
        )
    parser.add_argument('--logging_dir', type=str, default='logs', help=
        '[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***.'
        )
    parser.add_argument('--mixed_precision', type=str, default='no',
        choices=['no', 'fp16', 'bf16'], help=
        'Whether to use mixed precision. Choosebetween fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10.and an Nvidia Ampere GPU.'
        )
    parser.add_argument('--local_rank', type=int, default=-1, help=
        'For distributed training: local_rank')
    parser.add_argument('--checkpointing_steps', type=int, default=500,
        help=
        'Save a checkpoint of the training state every X updates. These checkpoints can be used both as final checkpoints in case they are better than the last checkpoint and are suitable for resuming training using `--resume_from_checkpoint`.'
        )
    parser.add_argument('--checkpoints_total_limit', type=int, default=None,
        help=
        'Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`. See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state for more docs'
        )
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
        help=
        'Whether training should be resumed from a previous checkpoint. Use a path saved by `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        )
    parser.add_argument('--enable_xformers_memory_efficient_attention',
        action='store_true', help='Whether or not to use xformers.')
    args = parser.parse_args()
    env_local_rank = int(os.environ.get('LOCAL_RANK', -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    if args.instance_data_dir is None:
        raise ValueError('You must specify a train data directory.')
    if args.with_prior_preservation:
        if args.class_data_dir is None:
            raise ValueError(
                'You must specify a data directory for class images.')
        if args.class_prompt is None:
            raise ValueError('You must specify prompt for class images.')
    return args
