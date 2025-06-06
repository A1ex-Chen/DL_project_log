def parse_args():
    parser = argparse.ArgumentParser(description=
        'Simple example of finetuning Würstchen Prior.')
    parser.add_argument('--rank', type=int, default=4, help=
        'The dimension of the LoRA update matrices.')
    parser.add_argument('--pretrained_decoder_model_name_or_path', type=str,
        default='warp-ai/wuerstchen', required=False, help=
        'Path to pretrained model or model identifier from huggingface.co/models.'
        )
    parser.add_argument('--pretrained_prior_model_name_or_path', type=str,
        default='warp-ai/wuerstchen-prior', required=False, help=
        'Path to pretrained model or model identifier from huggingface.co/models.'
        )
    parser.add_argument('--dataset_name', type=str, default=None, help=
        'The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private, dataset). It can also be a path pointing to a local copy of a dataset in your filesystem, or to a folder containing files that 🤗 Datasets can understand.'
        )
    parser.add_argument('--dataset_config_name', type=str, default=None,
        help=
        "The config of the Dataset, leave as None if there's only one config.")
    parser.add_argument('--train_data_dir', type=str, default=None, help=
        'A folder containing the training data. Folder contents must follow the structure described in https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file must exist to provide the captions for the images. Ignored if `dataset_name` is specified.'
        )
    parser.add_argument('--image_column', type=str, default='image', help=
        'The column of the dataset containing an image.')
    parser.add_argument('--caption_column', type=str, default='text', help=
        'The column of the dataset containing a caption or a list of captions.'
        )
    parser.add_argument('--max_train_samples', type=int, default=None, help
        =
        'For debugging purposes or quicker training, truncate the number of training examples to this value if set.'
        )
    parser.add_argument('--validation_prompts', type=str, default=None,
        nargs='+', help=
        'A set of prompts evaluated every `--validation_epochs` and logged to `--report_to`.'
        )
    parser.add_argument('--output_dir', type=str, default=
        'wuerstchen-model-finetuned-lora', help=
        'The output directory where the model predictions and checkpoints will be written.'
        )
    parser.add_argument('--cache_dir', type=str, default=None, help=
        'The directory where the downloaded models and datasets will be stored.'
        )
    parser.add_argument('--seed', type=int, default=None, help=
        'A seed for reproducible training.')
    parser.add_argument('--resolution', type=int, default=512, help=
        'The resolution for input images, all the images in the train/validation dataset will be resized to this resolution'
        )
    parser.add_argument('--train_batch_size', type=int, default=1, help=
        'Batch size (per device) for the training dataloader.')
    parser.add_argument('--num_train_epochs', type=int, default=100)
    parser.add_argument('--max_train_steps', type=int, default=None, help=
        'Total number of training steps to perform.  If provided, overrides num_train_epochs.'
        )
    parser.add_argument('--gradient_accumulation_steps', type=int, default=
        1, help=
        'Number of updates steps to accumulate before performing a backward/update pass.'
        )
    parser.add_argument('--learning_rate', type=float, default=0.0001, help
        ='learning rate')
    parser.add_argument('--lr_scheduler', type=str, default='constant',
        help=
        'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]'
        )
    parser.add_argument('--lr_warmup_steps', type=int, default=500, help=
        'Number of steps for the warmup in the lr scheduler.')
    parser.add_argument('--use_8bit_adam', action='store_true', help=
        'Whether or not to use 8-bit Adam from bitsandbytes.')
    parser.add_argument('--allow_tf32', action='store_true', help=
        'Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices'
        )
    parser.add_argument('--dataloader_num_workers', type=int, default=0,
        help=
        'Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.'
        )
    parser.add_argument('--adam_beta1', type=float, default=0.9, help=
        'The beta1 parameter for the Adam optimizer.')
    parser.add_argument('--adam_beta2', type=float, default=0.999, help=
        'The beta2 parameter for the Adam optimizer.')
    parser.add_argument('--adam_weight_decay', type=float, default=0.0,
        required=False, help='weight decay_to_use')
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
    parser.add_argument('--mixed_precision', type=str, default=None,
        choices=['no', 'fp16', 'bf16'], help=
        'Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config.'
        )
    parser.add_argument('--report_to', type=str, default='tensorboard',
        help=
        'The integration to report the results and logs to. Supported platforms are `"tensorboard"` (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        )
    parser.add_argument('--local_rank', type=int, default=-1, help=
        'For distributed training: local_rank')
    parser.add_argument('--checkpointing_steps', type=int, default=500,
        help=
        'Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming training using `--resume_from_checkpoint`.'
        )
    parser.add_argument('--checkpoints_total_limit', type=int, default=None,
        help='Max number of checkpoints to store.')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
        help=
        'Whether training should be resumed from a previous checkpoint. Use a path saved by `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        )
    parser.add_argument('--validation_epochs', type=int, default=5, help=
        'Run validation every X epochs.')
    parser.add_argument('--tracker_project_name', type=str, default=
        'text2image-fine-tune', help=
        'The `project_name` argument passed to Accelerator.init_trackers for more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator'
        )
    args = parser.parse_args()
    env_local_rank = int(os.environ.get('LOCAL_RANK', -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError('Need either a dataset name or a training folder.')
    return args
