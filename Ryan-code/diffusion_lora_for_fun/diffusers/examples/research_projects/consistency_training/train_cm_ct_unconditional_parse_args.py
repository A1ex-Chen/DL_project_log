def parse_args():
    parser = argparse.ArgumentParser(description=
        'Simple example of a training script.')
    parser.add_argument('--model_config_name_or_path', type=str, default=
        None, help=
        'The config of the UNet model to train, leave as None to use standard DDPM configuration.'
        )
    parser.add_argument('--pretrained_model_name_or_path', type=str,
        default=None, help=
        'If initializing the weights from a pretrained model, the path to the pretrained model or model identifier from huggingface.co/models.'
        )
    parser.add_argument('--revision', type=str, default=None, required=
        False, help=
        'Revision of pretrained model identifier from huggingface.co/models.')
    parser.add_argument('--variant', type=str, default=None, help=(
        'Variant of the model files of the pretrained model identifier from huggingface.co/models, e.g. `fp16`, `non_ema`, etc.'
        ,))
    parser.add_argument('--train_data_dir', type=str, default=None, help=
        'A folder containing the training data. Folder contents must follow the structure described in https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file must exist to provide the captions for the images. Ignored if `dataset_name` is specified.'
        )
    parser.add_argument('--dataset_name', type=str, default=None, help=
        'The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private, dataset). It can also be a path pointing to a local copy of a dataset in your filesystem, or to a folder containing files that HF Datasets can understand.'
        )
    parser.add_argument('--dataset_config_name', type=str, default=None,
        help=
        "The config of the Dataset, leave as None if there's only one config.")
    parser.add_argument('--dataset_image_column_name', type=str, default=
        'image', help=
        'The name of the image column in the dataset to use for training.')
    parser.add_argument('--dataset_class_label_column_name', type=str,
        default='label', help=
        'If doing class-conditional training, the name of the class label column in the dataset to use.'
        )
    parser.add_argument('--resolution', type=int, default=64, help=
        'The resolution for input images, all the images in the train/validation dataset will be resized to this resolution'
        )
    parser.add_argument('--interpolation_type', type=str, default=
        'bilinear', help=
        'The interpolation function used when resizing images to the desired resolution. Choose between `bilinear`, `bicubic`, `box`, `nearest`, `nearest_exact`, `hamming`, and `lanczos`.'
        )
    parser.add_argument('--center_crop', default=False, action='store_true',
        help=
        'Whether to center crop the input images to the resolution. If not set, the images will be randomly cropped. The images will be resized to the resolution first before cropping.'
        )
    parser.add_argument('--random_flip', default=False, action='store_true',
        help='whether to randomly flip images horizontally')
    parser.add_argument('--class_conditional', action='store_true', help=
        'Whether to train a class-conditional model. If set, the class labels will be taken from the `label` column of the provided dataset.'
        )
    parser.add_argument('--num_classes', type=int, default=None, help=
        'The number of classes in the training data, if training a class-conditional model.'
        )
    parser.add_argument('--class_embed_type', type=str, default=None, help=
        'The class embedding type to use. Choose from `None`, `identity`, and `timestep`. If `class_conditional` and `num_classes` and set, but `class_embed_type` is `None`, a embedding matrix will be used.'
        )
    parser.add_argument('--dataloader_num_workers', type=int, default=0,
        help=
        'The number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.'
        )
    parser.add_argument('--output_dir', type=str, default='ddpm-model-64',
        help=
        'The output directory where the model predictions and checkpoints will be written.'
        )
    parser.add_argument('--overwrite_output_dir', action='store_true')
    parser.add_argument('--cache_dir', type=str, default=None, help=
        'The directory where the downloaded models and datasets will be stored.'
        )
    parser.add_argument('--seed', type=int, default=None, help=
        'A seed for reproducible training.')
    parser.add_argument('--train_batch_size', type=int, default=16, help=
        'Batch size (per device) for the training dataloader.')
    parser.add_argument('--num_train_epochs', type=int, default=100)
    parser.add_argument('--max_train_steps', type=int, default=None, help=
        'Total number of training steps to perform.  If provided, overrides num_train_epochs.'
        )
    parser.add_argument('--max_train_samples', type=int, default=None, help
        =
        'For debugging purposes or quicker training, truncate the number of training examples to this value if set.'
        )
    parser.add_argument('--learning_rate', type=float, default=0.0001, help
        ='Initial learning rate (after the potential warmup period) to use.')
    parser.add_argument('--scale_lr', action='store_true', default=False,
        help=
        'Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.'
        )
    parser.add_argument('--lr_scheduler', type=str, default='cosine', help=
        'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]'
        )
    parser.add_argument('--lr_warmup_steps', type=int, default=500, help=
        'Number of steps for the warmup in the lr scheduler.')
    parser.add_argument('--optimizer_type', type=str, default='adamw', help
        =
        'The optimizer algorithm to use for training. Choose between `radam` and `adamw`. The iCT paper uses RAdam.'
        )
    parser.add_argument('--use_8bit_adam', action='store_true', help=
        'Whether or not to use 8-bit Adam from bitsandbytes.')
    parser.add_argument('--adam_beta1', type=float, default=0.95, help=
        'The beta1 parameter for the Adam optimizer.')
    parser.add_argument('--adam_beta2', type=float, default=0.999, help=
        'The beta2 parameter for the Adam optimizer.')
    parser.add_argument('--adam_weight_decay', type=float, default=1e-06,
        help='Weight decay magnitude for the Adam optimizer.')
    parser.add_argument('--adam_epsilon', type=float, default=1e-08, help=
        'Epsilon value for the Adam optimizer.')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, help=
        'Max gradient norm.')
    parser.add_argument('--prediction_type', type=str, default='sample',
        choices=['sample'], help=
        "Whether the model should predict the 'epsilon'/noise error or directly the reconstructed image 'x0'."
        )
    parser.add_argument('--ddpm_num_steps', type=int, default=1000)
    parser.add_argument('--ddpm_num_inference_steps', type=int, default=1000)
    parser.add_argument('--ddpm_beta_schedule', type=str, default='linear')
    parser.add_argument('--sigma_min', type=float, default=0.002, help=
        'The lower boundary for the timestep discretization, which should be set to a small positive value close to zero to avoid numerical issues when solving the PF-ODE backwards in time.'
        )
    parser.add_argument('--sigma_max', type=float, default=80.0, help=
        'The upper boundary for the timestep discretization, which also determines the variance of the Gaussian prior.'
        )
    parser.add_argument('--rho', type=float, default=7.0, help=
        'The rho parameter for the Karras sigmas timestep dicretization.')
    parser.add_argument('--huber_c', type=float, default=None, help=
        'The Pseudo-Huber loss parameter c. If not set, this will default to the value recommended in the Improved Consistency Training (iCT) paper of 0.00054 * sqrt(d), where d is the data dimensionality.'
        )
    parser.add_argument('--discretization_s_0', type=int, default=10, help=
        'The s_0 parameter in the discretization curriculum N(k). This controls the number of training steps after which the number of discretization steps N will be doubled.'
        )
    parser.add_argument('--discretization_s_1', type=int, default=1280,
        help=
        'The s_1 parameter in the discretization curriculum N(k). This controls the upper limit to the number of discretization steps used. Increasing this value will reduce the bias at the cost of higher variance.'
        )
    parser.add_argument('--constant_discretization_steps', action=
        'store_true', help=
        'Whether to set the discretization curriculum N(k) to be the constant value `discretization_s_0 + 1`. This is useful for testing when `max_number_steps` is small, when `k_prime` would otherwise be 0, causing a divide-by-zero error.'
        )
    parser.add_argument('--p_mean', type=float, default=-1.1, help=
        'The mean parameter P_mean for the (discretized) lognormal noise schedule, which controls the probability of sampling a (discrete) noise level sigma_i.'
        )
    parser.add_argument('--p_std', type=float, default=2.0, help=
        'The standard deviation parameter P_std for the (discretized) noise schedule, which controls the probability of sampling a (discrete) noise level sigma_i.'
        )
    parser.add_argument('--noise_precond_type', type=str, default='cm',
        help=
        'The noise preconditioning function to use for transforming the raw Karras sigmas into the timestep argument of the U-Net. Choose between `none` (the identity function), `edm`, and `cm`.'
        )
    parser.add_argument('--input_precond_type', type=str, default='cm',
        help=
        'The input preconditioning function to use for scaling the image input of the U-Net. Choose between `none` (a scaling factor of 1) and `cm`.'
        )
    parser.add_argument('--skip_steps', type=int, default=1, help=
        'The gap in indices between the student and teacher noise levels. In the iCT paper this is always set to 1, but theoretically this could be greater than 1 and/or altered according to a curriculum throughout training, much like the number of discretization steps is.'
        )
    parser.add_argument('--cast_teacher', action='store_true', help=
        'Whether to cast the teacher U-Net model to `weight_dtype` or leave it in full precision.'
        )
    parser.add_argument('--use_ema', action='store_true', help=
        'Whether to use Exponential Moving Average for the final model weights.'
        )
    parser.add_argument('--ema_min_decay', type=float, default=None, help=
        'The minimum decay magnitude for EMA. If not set, this will default to the value of `ema_max_decay`, resulting in a constant EMA decay rate.'
        )
    parser.add_argument('--ema_max_decay', type=float, default=0.99993,
        help=
        'The maximum decay magnitude for EMA. Setting `ema_min_decay` equal to this value will result in a constant decay rate.'
        )
    parser.add_argument('--use_ema_warmup', action='store_true', help=
        'Whether to use EMA warmup.')
    parser.add_argument('--ema_inv_gamma', type=float, default=1.0, help=
        'The inverse gamma value for the EMA decay.')
    parser.add_argument('--ema_power', type=float, default=3 / 4, help=
        'The power value for the EMA decay.')
    parser.add_argument('--mixed_precision', type=str, default='no',
        choices=['no', 'fp16', 'bf16'], help=
        'Whether to use mixed precision. Choosebetween fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10.and an Nvidia Ampere GPU.'
        )
    parser.add_argument('--allow_tf32', action='store_true', help=
        'Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices'
        )
    parser.add_argument('--gradient_checkpointing', action='store_true',
        help=
        'Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.'
        )
    parser.add_argument('--gradient_accumulation_steps', type=int, default=
        1, help=
        'Number of updates steps to accumulate before performing a backward/update pass.'
        )
    parser.add_argument('--enable_xformers_memory_efficient_attention',
        action='store_true', help='Whether or not to use xformers.')
    parser.add_argument('--local_rank', type=int, default=-1, help=
        'For distributed training: local_rank')
    parser.add_argument('--validation_steps', type=int, default=200, help=
        'Run validation every X steps.')
    parser.add_argument('--eval_batch_size', type=int, default=16, help=
        'The number of images to generate for evaluation. Note that if `class_conditional` and `num_classes` is set the effective number of images generated per evaluation step is `eval_batch_size * num_classes`.'
        )
    parser.add_argument('--save_images_epochs', type=int, default=10, help=
        'How often to save images during training.')
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
    parser.add_argument('--save_model_epochs', type=int, default=10, help=
        'How often to save the model during training.')
    parser.add_argument('--report_to', type=str, default='tensorboard',
        help=
        'The integration to report the results and logs to. Supported platforms are `"tensorboard"` (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        )
    parser.add_argument('--logging_dir', type=str, default='logs', help=
        '[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***.'
        )
    parser.add_argument('--push_to_hub', action='store_true', help=
        'Whether or not to push the model to the Hub.')
    parser.add_argument('--hub_token', type=str, default=None, help=
        'The token to use to push to the Model Hub.')
    parser.add_argument('--hub_model_id', type=str, default=None, help=
        'The name of the repository to keep in sync with the local `output_dir`.'
        )
    parser.add_argument('--hub_private_repo', action='store_true', help=
        'Whether or not to create a private repository.')
    parser.add_argument('--tracker_project_name', type=str, default=
        'consistency-training', help=
        'The `project_name` argument passed to Accelerator.init_trackers for more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator'
        )
    args = parser.parse_args()
    env_local_rank = int(os.environ.get('LOCAL_RANK', -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError(
            'You must specify either a dataset name from the hub or a train data directory.'
            )
    return args
