def parse_args():
    parser = argparse.ArgumentParser(description=
        'Finetune a diffusion model for text to audio generation task.')
    parser.add_argument('--stage', type=int, choices=[1, 2], default=2,
        help=
        'Specifies the stage of the disillation. Must be 1 or 2. Stage 2 corresponds to consistency distillation'
        )
    parser.add_argument('--train_file', type=str, default=
        'data/train_audiocaps.json', help=
        'A csv or a json file containing the training data.')
    parser.add_argument('--use_bf16', action='store_true', default=False,
        help='Use bf16 mixed precision training.')
    parser.add_argument('--use_lora', action='store_true', default=False,
        help='Use low-rank adaptation.')
    parser.add_argument('--validation_file', type=str, default=
        'data/valid_audiocaps.json', help=
        'A csv or a json file containing the validation data.')
    parser.add_argument('--test_file', type=str, default=
        'data/test_audiocaps_subset.json', help=
        'A csv or a json file containing the test data for generation.')
    parser.add_argument('--num_examples', type=int, default=-1, help=
        'How many examples to use for training and validation.')
    parser.add_argument('--text_encoder_name', type=str, default=
        'google/flan-t5-large', help=
        'Text encoder identifier from huggingface.co/models.')
    parser.add_argument('--scheduler_name', type=str, default=
        'stabilityai/stable-diffusion-2-1', help='Scheduler identifier.')
    parser.add_argument('--unet_model_name', type=str, default=None, help=
        'UNet model identifier from huggingface.co/models.')
    parser.add_argument('--unet_model_config', type=str, default=None, help
        ='UNet model config json path.')
    parser.add_argument('--tango_model', type=str, default=None, help=
        'Tango model identifier from huggingface: declare-lab/tango')
    parser.add_argument('--stage1_model', type=str, default=None, help=
        'Path to the stage-one pretrained model. This effective only when stage is 2.'
        )
    parser.add_argument('--snr_gamma', type=float, default=None, help=
        'SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. Default to None. More details here: https://arxiv.org/abs/2303.09556.'
        )
    parser.add_argument('--loss_type', type=str, default='mse', choices=[
        'mse', 'mel', 'stft', 'clap'], help=
        "Loss type. Must be one of ['mse', 'mel', 'stft', 'clap']. This effective only when stage is 2."
        )
    parser.add_argument('--finetune_vae', action='store_true', default=
        False, help='Unfreeze the VAE parameters. Default is False.')
    parser.add_argument('--freeze_text_encoder', action='store_true',
        default=False, help='Freeze the text encoder model.')
    parser.add_argument('--text_column', type=str, default='captions', help
        ='The name of the column in the datasets containing the input texts.')
    parser.add_argument('--audio_column', type=str, default='location',
        help=
        'The name of the column in the datasets containing the audio paths.')
    parser.add_argument('--augment', action='store_true', default=False,
        help='Augment training data.')
    parser.add_argument('--uncondition', action='store_true', default=False,
        help=
        '10% uncondition for training. Only applies to consistency distillation.'
        )
    parser.add_argument('--use_edm', action='store_true', default=False,
        help=
        'Use the Heun solver proposed in EDM. Only applies to consistency distillation.'
        )
    parser.add_argument('--use_karras', action='store_true', default=False,
        help=
        'Use the noise schedule proposed in EDM. Only effective when use_edm is True.'
        )
    parser.add_argument('--prefix', type=str, default=None, help=
        'Add prefix in text prompts.')
    parser.add_argument('--per_device_train_batch_size', type=int, default=
        2, help='Batch size (per device) for the training dataloader.')
    parser.add_argument('--per_device_eval_batch_size', type=int, default=2,
        help='Batch size (per device) for the validation dataloader.')
    parser.add_argument('--learning_rate', type=float, default=3e-05, help=
        'Initial learning rate (after the potential warmup period) to use.')
    parser.add_argument('--num_train_epochs', type=int, default=40, help=
        'Total number of training epochs to perform.')
    parser.add_argument('--max_train_steps', type=int, default=None, help=
        'Total number of training steps to perform. If provided, overrides num_train_epochs.'
        )
    parser.add_argument('--gradient_accumulation_steps', type=int, default=
        4, help=
        'Number of updates steps to accumulate before performing a backward/update pass.'
        )
    parser.add_argument('--lr_scheduler_type', type=SchedulerType, default=
        'linear', help='The scheduler type to use.', choices=['linear',
        'cosine', 'cosine_with_restarts', 'polynomial', 'constant',
        'constant_with_warmup'])
    parser.add_argument('--num_warmup_steps', type=int, default=0, help=
        'Number of steps for the warmup in the lr scheduler.')
    parser.add_argument('--adam_beta1', type=float, default=0.9, help=
        'The beta1 parameter for the Adam optimizer.')
    parser.add_argument('--adam_beta2', type=float, default=0.999, help=
        'The beta2 parameter for the Adam optimizer.')
    parser.add_argument('--adam_weight_decay', type=float, default=0.01,
        help='Weight decay to use.')
    parser.add_argument('--adam_epsilon', type=float, default=1e-08, help=
        'Epsilon value for the Adam optimizer.')
    parser.add_argument('--target_ema_decay', type=float, default=0.95,
        help=
        'Target network (for consistency ditillation) EMA decay rate. Default is 0.95.'
        )
    parser.add_argument('--ema_decay', type=float, default=0.999, help=
        'Exponential Model Average decay rate. Default is 0.999.')
    parser.add_argument('--num_diffusion_steps', type=int, default=18, help
        =
        'Number of diffusion steps for the teacher model. Only applies to consistency distillation.'
        )
    parser.add_argument('--teacher_guidance_scale', type=int, default=1,
        help=
        'The scale of classifier-free guidance used for the teacher model. If -1, then use random guidance scale drawn from Unif(0, 6).'
        )
    parser.add_argument('--output_dir', type=str, default=None, help=
        'Where to store the final model.')
    parser.add_argument('--seed', type=int, default=None, help=
        'A seed for reproducible training.')
    parser.add_argument('--checkpointing_steps', type=str, default='best',
        help=
        "Whether the various states should be saved at the end of every 'epoch' or 'best' whenever validation loss decreases."
        )
    parser.add_argument('--save_every', type=int, default=5, help=
        'Save model after every how many epochs when checkpointing_steps is `best`.'
        )
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
        help='If the training should continue from a local checkpoint folder.')
    parser.add_argument('--starting_epoch', type=int, default=0, help=
        'The starting epoch (useful when resuming from checkpoint)')
    parser.add_argument('--eval_first', action='store_true', help=
        'Whether to perform evaluation first before start training.')
    parser.add_argument('--with_tracking', action='store_true', help=
        'Whether to enable experiment trackers for logging.')
    parser.add_argument('--report_to', type=str, default='all', help=
        "The integration to report the results and logs to. Supported platforms are `'tensorboard'`, `'wandb'`, `'comet_ml'` and `'clearml'`.Use `'all'` (default) to report to all integrations.Only applicable when `--with_tracking` is passed."
        )
    args = parser.parse_args()
    if args.train_file is None and args.validation_file is None:
        raise ValueError('Need a training/validation file.')
    else:
        if args.train_file is not None:
            extension = args.train_file.split('.')[-1]
            assert extension in ['csv', 'json'
                ], '`train_file` should be a csv or a json file.'
        if args.validation_file is not None:
            extension = args.validation_file.split('.')[-1]
            assert extension in ['csv', 'json'
                ], '`validation_file` should be a csv or a json file.'
    return args
