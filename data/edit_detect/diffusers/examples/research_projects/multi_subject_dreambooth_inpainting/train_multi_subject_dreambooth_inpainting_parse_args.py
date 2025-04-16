def parse_args():
    parser = argparse.ArgumentParser(description=
        'Simple example of a training script.')
    parser.add_argument('--pretrained_model_name_or_path', type=str,
        default=None, required=True, help=
        'Path to pretrained model or model identifier from huggingface.co/models.'
        )
    parser.add_argument('--instance_data_dir', nargs='+', help=
        'Instance data directories')
    parser.add_argument('--output_dir', type=str, default=
        'text-inversion-model', help=
        'The output directory where the model predictions and checkpoints will be written.'
        )
    parser.add_argument('--seed', type=int, default=None, help=
        'A seed for reproducible training.')
    parser.add_argument('--resolution', type=int, default=512, help=
        'The resolution for input images, all the images in the train/validation dataset will be resized to this resolution'
        )
    parser.add_argument('--train_text_encoder', default=False, action=
        'store_true', help='Whether to train the text encoder')
    parser.add_argument('--train_batch_size', type=int, default=4, help=
        'Batch size (per device) for the training dataloader.')
    parser.add_argument('--sample_batch_size', type=int, default=4, help=
        'Batch size (per device) for sampling images.')
    parser.add_argument('--max_train_steps', type=int, default=None, help=
        'Total number of training steps to perform.  If provided, overrides num_train_epochs.'
        )
    parser.add_argument('--gradient_accumulation_steps', type=int, default=
        1, help=
        'Number of updates steps to accumulate before performing a backward/update pass.'
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
    parser.add_argument('--logging_dir', type=str, default='logs', help=
        '[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***.'
        )
    parser.add_argument('--mixed_precision', type=str, default='no',
        choices=['no', 'fp16', 'bf16'], help=
        'Whether to use mixed precision. Choosebetween fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10.and an Nvidia Ampere GPU.'
        )
    parser.add_argument('--checkpointing_steps', type=int, default=1000,
        help=
        'Save a checkpoint of the training state every X updates. These checkpoints can be used both as final checkpoints in case they are better than the last checkpoint and are suitable for resuming training using `--resume_from_checkpoint`.'
        )
    parser.add_argument('--checkpointing_from', type=int, default=1000,
        help='Start to checkpoint from step')
    parser.add_argument('--validation_steps', type=int, default=50, help=
        'Run validation every X steps. Validation consists of running the prompt `args.validation_prompt` multiple times: `args.num_validation_images` and logging the images.'
        )
    parser.add_argument('--validation_from', type=int, default=0, help=
        'Start to validate from step')
    parser.add_argument('--checkpoints_total_limit', type=int, default=None,
        help=
        'Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`. See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state for more docs'
        )
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
        help=
        'Whether training should be resumed from a previous checkpoint. Use a path saved by `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        )
    parser.add_argument('--validation_project_name', type=str, default=None,
        help='The w&b name.')
    parser.add_argument('--report_to_wandb', default=False, action=
        'store_true', help='Whether to report to weights and biases')
    args = parser.parse_args()
    return args
