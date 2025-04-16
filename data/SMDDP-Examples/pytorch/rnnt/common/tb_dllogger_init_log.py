def init_log(args):
    enabled = not dist.is_initialized() or dist.get_rank() == 0
    if enabled:
        fpath = args.log_file or os.path.join(args.output_dir, 'nvlog.json')
        backends = [JSONStreamBackend(Verbosity.DEFAULT, unique_log_fpath(
            fpath)), StdOutBackend(Verbosity.VERBOSE, step_format=
            stdout_step_format, metric_format=stdout_metric_format)]
    else:
        backends = []
    dllogger.init(backends=backends)
    dllogger.metadata('train_lrate', {'name': 'lrate', 'format': ':>3.2e'})
    for id_, pref in [('train', ''), ('train_avg', 'avg train '), (
        'dev_ema', '  dev ema ')]:
        dllogger.metadata(f'{id_}_loss', {'name': f'{pref}loss', 'format':
            ':>7.2f'})
        dllogger.metadata(f'{id_}_wer', {'name': f'{pref}wer', 'format':
            ':>6.2f'})
        dllogger.metadata(f'{id_}_pplx', {'name': f'{pref}pplx', 'format':
            ':>6.2f'})
        dllogger.metadata(f'{id_}_throughput', {'name': f'{pref}utts/s',
            'format': ':>5.0f'})
        dllogger.metadata(f'{id_}_took', {'name': 'took', 'unit': 's',
            'format': ':>5.2f'})
    tb_subsets = ['train', 'dev_ema']
    global tb_loggers
    tb_loggers = {s: TBLogger(enabled, args.output_dir, name=s) for s in
        tb_subsets}
    log_parameters(vars(args), tb_subset='train')
