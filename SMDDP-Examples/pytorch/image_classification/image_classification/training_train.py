def train(train_loader, model_and_loss, optimizer, scaler, lr_scheduler,
    logger, epoch, steps_per_epoch, timeout_handler, ema=None, use_amp=
    False, prof=-1, batch_size_multiplier=1, register_metrics=True):
    interrupted = False
    if register_metrics and logger is not None:
        logger.register_metric('train.loss', log.LOSS_METER(), verbosity=
            dllogger.Verbosity.DEFAULT, metadata=LOSS_METADATA)
        logger.register_metric('train.compute_ips', log.PERF_METER(),
            verbosity=dllogger.Verbosity.VERBOSE, metadata=IPS_METADATA)
        logger.register_metric('train.total_ips', log.PERF_METER(),
            verbosity=dllogger.Verbosity.DEFAULT, metadata=IPS_METADATA)
        logger.register_metric('train.data_time', log.PERF_METER(),
            verbosity=dllogger.Verbosity.VERBOSE, metadata=TIME_METADATA)
        logger.register_metric('train.compute_time', log.PERF_METER(),
            verbosity=dllogger.Verbosity.VERBOSE, metadata=TIME_METADATA)
    step = get_train_step(model_and_loss, optimizer, scaler=scaler, use_amp
        =use_amp, batch_size_multiplier=batch_size_multiplier)
    model_and_loss.train()
    end = time.time()
    optimizer.zero_grad()
    data_iter = enumerate(train_loader)
    if logger is not None:
        data_iter = logger.iteration_generator_wrapper(data_iter, mode='train')
    for i, (input, target) in data_iter:
        bs = input.size(0)
        lr_scheduler(optimizer, i, epoch)
        data_time = time.time() - end
        optimizer_step = (i + 1) % batch_size_multiplier == 0
        loss = step(input, target, optimizer_step=optimizer_step)
        if ema is not None:
            ema(model_and_loss, epoch * steps_per_epoch + i)
        it_time = time.time() - end
        if logger is not None:
            logger.log_metric('train.loss', loss.item(), bs)
            logger.log_metric('train.compute_ips', utils.calc_ips(bs, 
                it_time - data_time))
            logger.log_metric('train.total_ips', utils.calc_ips(bs, it_time))
            logger.log_metric('train.data_time', data_time)
            logger.log_metric('train.compute_time', it_time - data_time)
        end = time.time()
        if prof > 0 and i + 1 >= prof:
            time.sleep(5)
            break
        if (i + 1) % 20 == 0 and timeout_handler.interrupted:
            time.sleep(5)
            interrupted = True
            break
    return interrupted
