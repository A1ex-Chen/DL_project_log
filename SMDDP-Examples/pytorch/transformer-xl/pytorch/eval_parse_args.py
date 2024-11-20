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
                'eval']
    else:
        config = {}
    parser.add_argument('--work_dir', default='LM-TFM', type=str, help=
        'experiment directory')
    parser.add_argument('--debug', action='store_true', help=
        'run in debug mode (do not create exp dir)')
    parser.add_argument('--data', type=str, default='../data/wikitext-103',
        help='location of the data corpus')
    parser.add_argument('--manual', type=str, default=None, nargs='+', help
        ='run model on raw input data')
    parser.add_argument('--dataset', type=str, default='wt103', choices=[
        'wt103', 'lm1b', 'enwik8', 'text8'], help='dataset name')
    parser.add_argument('--split', type=str, default='all', choices=['all',
        'valid', 'test'], help='which split to evaluate')
    parser.add_argument('--affinity', type=str, default='single_unique',
        choices=['socket', 'single', 'single_unique',
        'socket_unique_interleaved', 'socket_unique_continuous', 'disabled'
        ], help='type of CPU affinity')
    parser.add_argument('--profile', action='store_true', help=
        'Enable profiling with DLProf')
    parser.add_argument('--type', type=str, default='pytorch', choices=[
        'pytorch', 'torchscript'], help='type of runtime to use')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size'
        )
    parser.add_argument('--tgt_len', type=int, default=64, help=
        'number of tokens to predict')
    parser.add_argument('--ext_len', type=int, default=0, help=
        'length of the extended context')
    parser.add_argument('--mem_len', type=int, default=640, help=
        'length of the retained previous heads')
    parser.add_argument('--seed', type=int, default=1111, help='Random seed')
    parser.add_argument('--clamp_len', type=int, default=-1, help=
        'max positional embedding index')
    parser.add_argument('--cuda', action='store_true', help=
        'Run evaluation on a GPU using CUDA')
    parser.add_argument('--model', type=str, default='', help=
        'path to the checkpoint')
    parser.add_argument('--manual_config', type=json.loads, default=None,
        help='Manually specify config for the model')
    parser.add_argument('--manual_vocab', type=str, default='word', choices
        =['word', 'bpe'], help='Manually specify type of vocabulary')
    parser.add_argument('--fp16', action='store_true', help=
        'Run training in fp16/mixed precision')
    parser.add_argument('--log_all_ranks', action='store_true', help=
        'Enable logging for all distributed ranks')
    parser.add_argument('--dllog_file', type=str, default='eval_log.json',
        help='Name of the DLLogger output file')
    parser.add_argument('--same_length', action='store_true', help=
        'set same length attention with masking')
    parser.add_argument('--no_env', action='store_true', help=
        'Do not print info on execution env')
    parser.add_argument('--log_interval', type=int, default=10, help=
        'Report interval')
    parser.add_argument('--target_perplexity', type=float, default=None,
        help='target perplexity')
    parser.add_argument('--target_throughput', type=float, default=None,
        help='target throughput')
    parser.add_argument('--save_data', action='store_true', help=
        'save latency and throughput data to a file')
    parser.add_argument('--repeat', type=int, default=1, help=
        'loop over the dataset REPEAT times')
    parser.add_argument('--max_size', type=int, default=None, help=
        'run inference on up to MAX_SIZE batches')
    parser.add_argument('--percentiles', nargs='+', default=[90, 95, 99],
        help='percentiles for latency confidence intervals')
    parser.add_argument('--save_torchscript', default=None, type=str, help=
        'save torchscript model to a file')
    parser.add_argument('--load_torchscript', default=None, type=str, help=
        'load torchscript model from a file')
    parser.add_argument('--local_rank', type=int, default=os.getenv(
        'LOCAL_RANK', 0), help='Used for multi-process training.')
    parser.set_defaults(**config)
    args, _ = parser.parse_known_args()
    if args.manual:
        args.batch_size = 1
    if args.same_length and args.tgt_len > args.mem_len:
        warnings.warn(
            '--same_length is intended to be used with large mem_len relative to tgt_len'
            )
    if args.ext_len < 0:
        raise RuntimeError('Extended context length must be non-negative')
    return args
