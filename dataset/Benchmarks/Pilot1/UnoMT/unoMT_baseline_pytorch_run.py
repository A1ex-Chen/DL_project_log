def run(params):
    args = candle.ArgumentStruct(**params)
    seed = args.rng_seed
    candle.set_seed(seed)
    seed_random_state(args.rng_seed)
    if params['epochs'] < params['resp_val_start_epoch']:
        raise Exception(
            'Number of epochs is less than validation threshold (resp_val_start_epoch)'
            )
    now = datetime.datetime.now()
    ext = '%02d%02d_%02d%02d_pytorch' % (now.month, now.day, now.hour, now.
        minute)
    candle.verify_path(params['save_path'])
    prefix = '{}{}'.format(params['save_path'], ext)
    logfile = params['logfile'] if params['logfile'] else prefix + '.log'
    candle.set_up_logger(logfile, unoMT.logger, params['verbose'])
    unoMT.logger.info('Params: {}'.format(params))
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    modelUno = UnoMTModel(args, use_cuda, device)
    modelUno.pre_train_config()
    modelUno.train()
    modelUno.print_final_stats()
