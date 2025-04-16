def restart(gParameters, model, verbose=True):
    """
    Possibly restarts model from CheckpointCallback according to given
    settings and the ckpt-info.json

    return
           The JSON dict if the restart happened or
           None if the restart did not happen.
    """
    import logging
    logger = logging.getLogger('Candle.restart')
    directory = param(gParameters, 'ckpt_directory', './save')
    set_up_logger(directory + '/ckpt.log', logger, verbose=verbose,
        fmt_line='%(asctime)s CANDLE restart(): %(message)s')
    param_ckpt_mode = param(gParameters, 'ckpt_restart_mode', 'auto',
        allowed=['off', 'auto', 'required'])
    if param_ckpt_mode == 'off':
        return None
    dir_last = 'save/ckpts/last'
    model_file = dir_last + '/model.h5'
    if not os.path.exists(model_file):
        if param_ckpt_mode == 'required':
            raise Exception(
                "ckpt_restart_mode=='required' but no checkpoint " +
                'could be found!')
        assert param_ckpt_mode == 'auto'
        return None
    logger.info("restarting: '%s'" % model_file)
    result = restart_json(gParameters, logger, dir_last)
    logger.info('restarting: epoch=%i timestamp=%s', result['epoch'],
        result['timestamp'])
    start = time.time()
    stats = os.stat(model_file)
    MB = stats.st_size / (1024 * 1024)
    model.load_weights(model_file)
    stop = time.time()
    duration = stop - start
    rate = MB / duration
    logger.info(
        'restarting: model read:  %0.3f MB in %0.3f seconds (%0.2f MB/s).',
        MB, duration, rate)
    return result
