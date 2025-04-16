def parse_args():
    parser = argparse.ArgumentParser(description='RNN-T Training Reference')
    training = parser.add_argument_group('training setup')
    training.add_argument('--epochs', default=100, type=int, help=
        'Number of epochs for the entire training')
    training.add_argument('--warmup_epochs', default=6, type=int, help=
        'Initial epochs of increasing learning rate')
    training.add_argument('--hold_epochs', default=40, type=int, help=
        'Constant max learning rate epochs after warmup')
    training.add_argument('--epochs_this_job', default=0, type=int, help=
        'Run for a number of epochs with no effect on the lr schedule.Useful for re-starting the training.'
        )
    training.add_argument('--cudnn_benchmark', action='store_true', default
        =True, help='Enable cudnn benchmark')
    training.add_argument('--amp_level', default=1, type=int, choices=[0, 1,
        2, 3], help='APEX AMP optimization level')
    training.add_argument('--seed', default=None, type=int, help='Random seed')
    training.add_argument('--local_rank', default=dist.get_local_rank(),
        type=int, help='GPU id used for distributed training')
    training.add_argument('--target', default=0.058, type=float, help=
        'Target WER accuracy')
    training.add_argument('--apex_transducer_loss', default=None, type=str,
        choices=['fp16', 'fp32'], help=
        'what precision of apex transducer_loss to use')
    training.add_argument('--fuse_relu_dropout', action='store_true', help=
        'Fuse ReLU and dropout in the joint net')
    training.add_argument('--weights_init_scale', default=0.5, type=float,
        help='If set, overwrites value in config.')
    training.add_argument('--hidden_hidden_bias_scale', type=float, help=
        'If set, overwrites value in config.')
    training.add_argument('--batch_eval_mode', default=None, type=str,
        choices=['no_cg', 'cg', 'cg_unroll_pipeline'], help=
        'do evaluation in batch')
    training.add_argument('--cg_unroll_factor', default=4, type=int, help=
        'Unrolling factor for batch eval mode cg_unroll_pipeline')
    training.add_argument('--apex_transducer_joint', default=None, type=str,
        choices=['pack', 'not_pack'], help=
        'whether or not to pack the sequence with apex transducer_joint')
    training.add_argument('--buffer_pre_alloc', action='store_true', help=
        'Pre-allocate buffer in PyTorch')
    training.add_argument('--multilayer_lstm', action='store_true', help=
        'Use multilayer LSTMs instead of splitting them into multiple single-layer ones'
        )
    training.add_argument('--batch_split_factor', default=1, type=int, help
        ='Split batches feed into the joint net')
    training.add_argument('--apex_mlp', action='store_true', help=
        'Use apex MLP')
    training.add_argument('--num_cg', default=0, type=int, help=
        'number of graphs needed for training')
    training.add_argument('--min_seq_split_len', default=-1, type=int, help
        ='Split sequences in a mini-batch to improve performance')
    training.add_argument('--pre_sort_for_seq_split', action='store_true',
        help=
        'Presort samples in a mini-batch so that seq split is more effective')
    optim = parser.add_argument_group('optimization setup')
    optim.add_argument('--batch_size', default=128, type=int, help=
        'Effective batch size per GPU (might require grad accumulation')
    optim.add_argument('--val_batch_size', default=2, type=int, help=
        'Evalution time batch size')
    optim.add_argument('--lr', default=0.004, type=float, help=
        'Peak learning rate')
    optim.add_argument('--min_lr', default=1e-05, type=float, help=
        'minimum learning rate')
    optim.add_argument('--lr_exp_gamma', default=0.935, type=float, help=
        'gamma factor for exponential lr scheduler')
    optim.add_argument('--weight_decay', default=0.001, type=float, help=
        'Weight decay for the optimizer')
    optim.add_argument('--grad_accumulation_steps', default=8, type=int,
        help='Number of accumulation steps')
    optim.add_argument('--clip_norm', default=1, type=float, help=
        'If provided, gradients will be clipped above this norm')
    optim.add_argument('--beta1', default=0.9, type=float, help=
        'Beta 1 for optimizer')
    optim.add_argument('--beta2', default=0.999, type=float, help=
        'Beta 2 for optimizer')
    optim.add_argument('--ema', type=float, default=0.999, help=
        'Discount factor for exp averaging of model weights')
    optim.add_argument('--multi_tensor_ema', action='store_true', help=
        'Use multi_tensor_apply for EMA')
    optim.add_argument('--dist_lamb', action='store_true', help=
        'Use distributed LAMB')
    optim.add_argument('--ema_update_type', default='fp32', type=str,
        choices=['fp16', 'fp32'], help=
        'is ema applied on the fp32 master weight or fp16 weight')
    optim.add_argument('--dwu_group_size', default=8, type=int, help=
        'Group size for distributed optimizer. Will be ignored if non-distributed optimizer is used'
        )
    io = parser.add_argument_group('feature and checkpointing setup')
    io.add_argument('--dali_device', type=str, choices=['cpu', 'gpu'],
        default='cpu', help='Use DALI pipeline for fast data processing')
    io.add_argument('--resume', action='store_true', help=
        'Try to resume from last saved checkpoint.')
    io.add_argument('--ckpt', default=None, type=str, help=
        'Path to a checkpoint for resuming training')
    io.add_argument('--save_at_the_end', action='store_true', help=
        'Saves model checkpoint at the end of training')
    io.add_argument('--save_frequency', default=None, type=int, help=
        'Checkpoint saving frequency in epochs')
    io.add_argument('--keep_milestones', default=[], type=int, nargs='+',
        help='Milestone checkpoints to keep from removing')
    io.add_argument('--save_best_from', default=200, type=int, help=
        'Epoch on which to begin tracking best checkpoint (dev WER)')
    io.add_argument('--val_frequency', default=1, type=int, help=
        'Number of epochs between evaluations on dev set')
    io.add_argument('--log_frequency', default=25, type=int, help=
        'Number of steps between printing training stats')
    io.add_argument('--prediction_frequency', default=None, type=int, help=
        'Number of steps between printing sample decodings')
    io.add_argument('--model_config', default=
        'configs/baseline_v3-1023sp.yaml', type=str, required=True, help=
        'Path of the model configuration file')
    io.add_argument('--num_buckets', type=int, default=6, help=
        'If provided, samples will be grouped by audio duration, to this number of backets, for each bucket, random samples are batched, and finally all batches are randomly shuffled'
        )
    io.add_argument('--vectorized_sampler', action='store_true', help=
        'Use optimized bucketing sampler implementation')
    io.add_argument('--dist_sampler', action='store_true', help=
        'Each rank owns an unique copy of file list')
    io.add_argument('--train_manifests', type=str, required=True, nargs='+',
        help='Paths of the training dataset manifest file')
    io.add_argument('--val_manifests', type=str, required=True, nargs='+',
        help='Paths of the evaluation datasets manifest files')
    io.add_argument('--max_duration', type=float, help=
        'Discard samples longer than max_duration')
    io.add_argument('--max_txt_len', type=int, default=125, help=
        'The longest text length in the sample')
    io.add_argument('--max_eval_sample_duration', type=float, default=32.7,
        help='The max duration of samples in the eval set')
    io.add_argument('--dataset_dir', required=True, type=str, help=
        'Root dir of dataset')
    io.add_argument('--output_dir', type=str, required=True, help=
        'Directory for logs and checkpoints')
    io.add_argument('--log_file', type=str, default=None, help=
        'Path to save the training logfile.')
    io.add_argument('--max_symbol_per_sample', type=int, default=None, help
        ='maximum number of symbols per sample can have during eval')
    io.add_argument('--data_cpu_threads', type=int, default=multiprocessing
        .cpu_count(), help=
        'Number of CPU threads used for data loading and preprocessing.')
    io.add_argument('--synthetic_audio_seq_len', type=int, default=None,
        help='length for synthetic audio sequence.')
    io.add_argument('--synthetic_text_seq_len', type=int, default=None,
        help='length for synthetic text sequence.')
    io.add_argument('--enable_seq_len_stats', action='store_true', help=
        'Store and output seq len stats')
    io.add_argument('--vectorized_sa', action='store_true', help=
        'Vectorized implementation of SpecAugment')
    io.add_argument('--in_mem_file_list', action='store_true', help=
        'prepare file list in memory instead of on the disk')
    io.add_argument('--enable_prefetch', action='store_true', help=
        'prefetch and preprocess input data for next iteration')
    io.add_argument('--tokenized_transcript', action='store_true', help=
        'loads transcript in tokenized form')
    io.add_argument('--jit_tensor_formation', action='store_true', help=
        'just-in-time tensor formation. Form the input txt tensor on the fly.')
    io.add_argument('--dali_dont_use_mmap', action='store_true', help=
        'Disable mmap for DALI')
    return parser.parse_args()
