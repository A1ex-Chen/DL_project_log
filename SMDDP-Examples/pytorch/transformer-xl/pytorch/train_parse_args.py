def parse_args():
    parent_parser = argparse.ArgumentParser(description=
        'PyTorch Transformer-XL Language Model', formatter_class=argparse.
        ArgumentDefaultsHelpFormatter, add_help=False)
    parser = argparse.ArgumentParser(parents=[parent_parser], add_help=True)
    cfg_parser = argparse.ArgumentParser(parents=[parent_parser], add_help=
        False)
    cfg_parser.add_argument('--config', default='default')
    cfg_parser.add_argument('--config_file', default=None)
    config_args, _ = cfg_parser.parse_known_args()
    if config_args.config is not None and config_args.config_file is not None:
        with open(config_args.config_file) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)[config_args.config][
                'train']
    else:
        config = {}
    general = parser.add_argument_group('general setup')
    general.add_argument('--work_dir', default='LM-TFM', type=str, help=
        'Directory for the results')
    general.add_argument('--append_dataset', action='store_true', help=
        'Automatically append dataset name to work_dir')
    general.add_argument('--append_time', action='store_true', help=
        'Automatically append current time to work_dir')
    general.add_argument('--cuda', action='store_true', help=
        'Run training on a GPU using CUDA')
    general.add_argument('--fp16', action='store_true', help=
        'Run training in fp16/mixed precision')
    general.add_argument('--restart', type=str, default='', help=
        'Restart training from the saved checkpoint')
    general.add_argument('--debug', action='store_true', help=
        'Run in debug mode (do not create exp dir)')
    general.add_argument('--log_all_ranks', action='store_true', help=
        'Enable logging from all distributed ranks')
    general.add_argument('--dllog_file', type=str, default='train_log.json',
        help='Name of the DLLogger output file')
    general.add_argument('--txtlog_file', type=str, default='train_log.log',
        help='Name of the txt log file')
    general.add_argument('--save_all', action='store_true', help=
        'Save all checkpoints')
    general.add_argument('--no_env', action='store_true', help=
        'Do not print info on execution env')
    general.add_argument('--no_eval', action='store_true', help=
        'Disable model evaluation')
    general.add_argument('--log_interval', type=int, default=10, help=
        'Report interval')
    general.add_argument('--target_throughput', type=float, default=None,
        help='Target training throughput (for benchmarking)')
    general.add_argument('--target_perplexity', type=float, default=None,
        help='Target validation perplexity (for benchmarking)')
    general.add_argument('--apex_amp_opt_level', type=str, default='O2',
        choices=['O0', 'O1', 'O2', 'O3'], help=
        'Optimization level for apex amp')
    general.add_argument('--amp', choices=['apex', 'pytorch'], default=
        'apex', help='Implementation of automatic mixed precision')
    general.add_argument('--affinity', type=str, default=
        'socket_unique_interleaved', choices=['socket', 'single',
        'single_unique', 'socket_unique_interleaved',
        'socket_unique_continuous', 'disabled'], help='type of CPU affinity')
    general.add_argument('--profile', action='store_true', help=
        'Enable profiling with DLProf')
    dataset = parser.add_argument_group('dataset setup')
    dataset.add_argument('--data', type=str, default='../data/wikitext-103',
        help='Location of the data corpus')
    dataset.add_argument('--dataset', type=str, default='wt103', choices=[
        'wt103', 'lm1b', 'enwik8', 'text8'], help='Dataset name')
    dataset.add_argument('--vocab', type=str, default='word', choices=[
        'word', 'bpe'], help='Type of vocabulary')
    model = parser.add_argument_group('model setup')
    model.add_argument('--n_layer', type=int, default=16, help=
        'Number of total layers')
    model.add_argument('--n_head', type=int, default=8, help='Number of heads')
    model.add_argument('--d_head', type=int, default=64, help='Head dimension')
    model.add_argument('--d_embed', type=int, default=-1, help=
        'Embedding dimension')
    model.add_argument('--d_model', type=int, default=512, help=
        'Model dimension')
    model.add_argument('--d_inner', type=int, default=2048, help=
        'Inner dimension in feedforward layer')
    model.add_argument('--dropout', type=float, default=0.1, help=
        'Global dropout rate')
    model.add_argument('--dropatt', type=float, default=0.0, help=
        'Attention probability dropout rate')
    model.add_argument('--pre_lnorm', action='store_true', help=
        'Apply LayerNorm to the input instead of the output')
    model.add_argument('--attn_type', type=int, default=0, help=
        'Attention type. 0 for ours, 1 for Shaw et al,2 for Vaswani et al, 3 for Al Rfou et al.'
        )
    model.add_argument('--not_tied', action='store_true', help=
        'Do not tie the word embedding and softmax weights')
    model.add_argument('--clamp_len', type=int, default=-1, help=
        'Use the same pos embeddings after clamp_len')
    model.add_argument('--adaptive', action='store_true', help=
        'Use adaptive softmax')
    model.add_argument('--div_val', type=int, default=1, help=
        'Dividend value for adaptive input and softmax')
    model.add_argument('--sample_softmax', type=int, default=-1, help=
        'Number of samples in sampled softmax')
    model.add_argument('--init', default='normal', type=str, help=
        'Parameter initializer to use')
    model.add_argument('--emb_init', default='normal', type=str, help=
        'Parameter initializer to use')
    model.add_argument('--init_range', type=float, default=0.1, help=
        'Parameters initialized by U(-init_range, init_range)')
    model.add_argument('--emb_init_range', type=float, default=0.01, help=
        'Parameters initialized by U(-init_range, init_range)')
    model.add_argument('--init_std', type=float, default=0.02, help=
        'Parameters initialized by N(0, init_std)')
    model.add_argument('--proj_init_std', type=float, default=0.01, help=
        'Parameters initialized by N(0, init_std)')
    opt = parser.add_argument_group('optimizer setup')
    opt.add_argument('--optim', default='jitlamb', type=str, choices=[
        'adam', 'sgd', 'adagrad', 'lamb', 'jitlamb'], help='Optimizer to use')
    opt.add_argument('--lr', type=float, default=0.01, help=
        'Initial learning rate')
    opt.add_argument('--mom', type=float, default=0.0, help='Momentum for sgd')
    opt.add_argument('--scheduler', default='cosine', type=str, choices=[
        'cosine', 'inv_sqrt', 'dev_perf', 'constant'], help=
        'LR scheduler to use')
    opt.add_argument('--max_step_scheduler', type=int, default=None, help=
        'Max number of training steps for LR scheduler')
    opt.add_argument('--warmup_step', type=int, default=1000, help=
        'Number of iterations for LR warmup')
    opt.add_argument('--decay_rate', type=float, default=0.5, help=
        'Decay factor when ReduceLROnPlateau is used')
    opt.add_argument('--lr_min', type=float, default=0.0, help=
        'Minimum learning rate during annealing')
    opt.add_argument('--clip', type=float, default=0.25, help=
        'Gradient clipping')
    opt.add_argument('--weight_decay', type=float, default=0.0, help=
        'Weight decay for adam|lamb')
    opt.add_argument('--clip_nonemb', action='store_true', help=
        'Only clip the gradient of non-embedding params')
    opt.add_argument('--patience', type=int, default=0, help='Patience')
    opt.add_argument('--eta_min', type=float, default=0.001, help=
        'Min learning rate for cosine scheduler')
    training = parser.add_argument_group('training setup')
    training.add_argument('--max_step', type=int, default=40000, help=
        'Max number of training steps')
    training.add_argument('--batch_size', type=int, default=256, help=
        'Global batch size')
    training.add_argument('--local_batch_size', type=int, default=None,
        help=
        'Local (per-device) batch size, this setting                           overrides global --batch_size and sets batch_size                           to local_batch_size * world_size'
        )
    training.add_argument('--batch_chunk', type=int, default=1, help=
        'Split batch into chunks and train with gradient accumulation')
    training.add_argument('--roll', action='store_true', help=
        'Enable random shifts within each data stream')
    training.add_argument('--tgt_len', type=int, default=192, help=
        'Number of tokens to predict')
    training.add_argument('--ext_len', type=int, default=0, help=
        'Length of the extended context')
    training.add_argument('--mem_len', type=int, default=192, help=
        'Length of the retained previous heads')
    training.add_argument('--seed', type=int, default=1111, help='Random seed')
    training.add_argument('--multi_gpu', default=None, type=str, choices=[
        'ddp', 'dp'], help='Use multiple GPU')
    training.add_argument('--gpu0_bsz', type=int, default=-1, help=
        'Batch size on gpu 0 (for "dp" backend)')
    training.add_argument('--same_length', action='store_true', help=
        'Use the same attn length for all tokens')
    training.add_argument('--varlen', action='store_true', help=
        'Use variable length')
    training.add_argument('--swap_mem', action='store_true', help=
        'Swap memory tensors to cpu')
    val = parser.add_argument_group('validation setup')
    val.add_argument('--eval_tgt_len', type=int, default=192, help=
        'Number of tokens to predict for evaluation')
    val.add_argument('--eval_batch_size', type=int, default=16, help=
        'Eval batch size')
    val.add_argument('--eval_max_steps', type=int, default=-1, help=
        'Max eval steps')
    val.add_argument('--eval_interval', type=int, default=5000, help=
        'Evaluation interval')
    distr = parser.add_argument_group('distributed setup')
    distr.add_argument('--local_rank', type=int, default=dist.
        get_local_rank(), help='Used for multi-process training.')
    parser.set_defaults(**config)
    args, _ = parser.parse_known_args()
    args.tied = not args.not_tied
    if args.d_embed < 0:
        args.d_embed = args.d_model
    if args.ext_len < 0:
        raise RuntimeError('Extended context length must be non-negative')
    if args.mem_len == 0:
        if args.eval_tgt_len > args.ext_len + args.tgt_len:
            raise RuntimeError(
                f'eval_tgt_len should be <= tgt_len + ext_len; eval_tgt_len: {args.eval_tgt_len}, tgt_len: {args.tgt_len}, ext_len: {args.ext_len}'
                )
    elif args.eval_tgt_len > args.mem_len + args.tgt_len:
        raise RuntimeError(
            f'eval_tgt_len should be <= tgt_len + mem_len; eval_tgt_len: {args.eval_tgt_len}, tgt_len: {args.tgt_len}, mem_len: {args.mem_len}'
            )
    if args.batch_size % args.batch_chunk != 0:
        raise RuntimeError('Batch size needs to be divisible by batch chunk')
    if args.fp16 and args.amp == 'apex' and 'apex' not in sys.modules:
        raise RuntimeError(
            'APEX AMP unavailable, install APEX or switch to pytorch AMP')
    return args
