def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model_name_or_path', type=str,
        default=None, required=True, help=
        'Path to pretrained model or model identifier from huggingface.co/models.'
        )
    parser.add_argument('--revision', type=str, default=None, required=
        False, help=
        'Revision of pretrained model identifier from huggingface.co/models.')
    parser.add_argument('--variant', type=str, default=None, help=
        "Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16"
        )
    parser.add_argument('--instance_data_dataset', type=str, default=None,
        required=False, help=
        'A Hugging Face dataset containing the training images')
    parser.add_argument('--instance_data_dir', type=str, default=None,
        required=False, help=
        'A folder containing the training data of instance images.')
    parser.add_argument('--instance_data_image', type=str, default=None,
        required=False, help='A single training image')
    parser.add_argument('--use_8bit_adam', action='store_true', help=
        'Whether or not to use 8-bit Adam from bitsandbytes.')
    parser.add_argument('--dataloader_num_workers', type=int, default=0,
        help=
        'Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.'
        )
    parser.add_argument('--allow_tf32', action='store_true', help=
        'Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices'
        )
    parser.add_argument('--use_ema', action='store_true', help=
        'Whether to use EMA model.')
    parser.add_argument('--ema_decay', type=float, default=0.9999)
    parser.add_argument('--ema_update_after_step', type=int, default=0)
    parser.add_argument('--adam_beta1', type=float, default=0.9, help=
        'The beta1 parameter for the Adam optimizer.')
    parser.add_argument('--adam_beta2', type=float, default=0.999, help=
        'The beta2 parameter for the Adam optimizer.')
    parser.add_argument('--adam_weight_decay', type=float, default=0.01,
        help='Weight decay to use.')
    parser.add_argument('--adam_epsilon', type=float, default=1e-08, help=
        'Epsilon value for the Adam optimizer')
    parser.add_argument('--output_dir', type=str, default='muse_training',
        help=
        'The output directory where the model predictions and checkpoints will be written.'
        )
    parser.add_argument('--seed', type=int, default=None, help=
        'A seed for reproducible training.')
    parser.add_argument('--logging_dir', type=str, default='logs', help=
        '[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***.'
        )
    parser.add_argument('--max_train_steps', type=int, default=None, help=
        'Total number of training steps to perform.  If provided, overrides num_train_epochs.'
        )
    parser.add_argument('--checkpointing_steps', type=int, default=500,
        help=
        'Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference.Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components.See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by stepinstructions.'
        )
    parser.add_argument('--logging_steps', type=int, default=50)
    parser.add_argument('--checkpoints_total_limit', type=int, default=None,
        help=
        'Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`. See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state for more details'
        )
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
        help=
        'Whether training should be resumed from a previous checkpoint. Use a path saved by `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        )
    parser.add_argument('--train_batch_size', type=int, default=16, help=
        'Batch size (per device) for the training dataloader.')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=
        1, help=
        'Number of updates steps to accumulate before performing a backward/update pass.'
        )
    parser.add_argument('--learning_rate', type=float, default=0.0003, help
        ='Initial learning rate (after the potential warmup period) to use.')
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
    parser.add_argument('--validation_steps', type=int, default=100, help=
        'Run validation every X steps. Validation consists of running the prompt `args.validation_prompt` multiple times: `args.num_validation_images` and logging the images.'
        )
    parser.add_argument('--mixed_precision', type=str, default=None,
        choices=['no', 'fp16', 'bf16'], help=
        'Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config.'
        )
    parser.add_argument('--report_to', type=str, default='wandb', help=
        'The integration to report the results and logs to. Supported platforms are `"tensorboard"` (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        )
    parser.add_argument('--validation_prompts', type=str, nargs='*')
    parser.add_argument('--resolution', type=int, default=512, help=
        'The resolution for input images, all the images in the train/validation dataset will be resized to this resolution'
        )
    parser.add_argument('--split_vae_encode', type=int, required=False,
        default=None)
    parser.add_argument('--min_masking_rate', type=float, default=0.0)
    parser.add_argument('--cond_dropout_prob', type=float, default=0.0)
    parser.add_argument('--max_grad_norm', default=None, type=float, help=
        'Max gradient norm.', required=False)
    parser.add_argument('--use_lora', action='store_true', help=
        'Fine tune the model using LoRa')
    parser.add_argument('--text_encoder_use_lora', action='store_true',
        help='Fine tune the model using LoRa')
    parser.add_argument('--lora_r', default=16, type=int)
    parser.add_argument('--lora_alpha', default=32, type=int)
    parser.add_argument('--lora_target_modules', default=['to_q', 'to_k',
        'to_v'], type=str, nargs='+')
    parser.add_argument('--text_encoder_lora_r', default=16, type=int)
    parser.add_argument('--text_encoder_lora_alpha', default=32, type=int)
    parser.add_argument('--text_encoder_lora_target_modules', default=[
        'to_q', 'to_k', 'to_v'], type=str, nargs='+')
    parser.add_argument('--train_text_encoder', action='store_true')
    parser.add_argument('--image_key', type=str, required=False)
    parser.add_argument('--prompt_key', type=str, required=False)
    parser.add_argument('--gradient_checkpointing', action='store_true',
        help=
        'Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.'
        )
    parser.add_argument('--prompt_prefix', type=str, required=False,
        default=None)
    args = parser.parse_args()
    if args.report_to == 'wandb':
        if not is_wandb_available():
            raise ImportError(
                'Make sure to install wandb if you want to use it for logging during training.'
                )
    num_datasources = sum([(x is not None) for x in [args.instance_data_dir,
        args.instance_data_image, args.instance_data_dataset]])
    if num_datasources != 1:
        raise ValueError(
            'provide one and only one of `--instance_data_dir`, `--instance_data_image`, or `--instance_data_dataset`'
            )
    if args.instance_data_dir is not None:
        if not os.path.exists(args.instance_data_dir):
            raise ValueError(
                f'Does not exist: `--args.instance_data_dir` {args.instance_data_dir}'
                )
    if args.instance_data_image is not None:
        if not os.path.exists(args.instance_data_image):
            raise ValueError(
                f'Does not exist: `--args.instance_data_image` {args.instance_data_image}'
                )
    if args.instance_data_dataset is not None and (args.image_key is None or
        args.prompt_key is None):
        raise ValueError(
            '`--instance_data_dataset` requires setting `--image_key` and `--prompt_key`'
            )
    return args
