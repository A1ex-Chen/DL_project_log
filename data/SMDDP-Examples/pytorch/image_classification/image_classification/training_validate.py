def validate(val_loader, model_and_loss, logger, epoch, use_amp=False, prof
    =-1, register_metrics=True, prefix='val'):
    if register_metrics and logger is not None:
        logger.register_metric(f'{prefix}.top1', log.ACC_METER(), verbosity
            =dllogger.Verbosity.DEFAULT, metadata=ACC_METADATA)
        logger.register_metric(f'{prefix}.top5', log.ACC_METER(), verbosity
            =dllogger.Verbosity.DEFAULT, metadata=ACC_METADATA)
        logger.register_metric(f'{prefix}.loss', log.LOSS_METER(),
            verbosity=dllogger.Verbosity.DEFAULT, metadata=LOSS_METADATA)
        logger.register_metric(f'{prefix}.compute_ips', log.PERF_METER(),
            verbosity=dllogger.Verbosity.VERBOSE, metadata=IPS_METADATA)
        logger.register_metric(f'{prefix}.total_ips', log.PERF_METER(),
            verbosity=dllogger.Verbosity.DEFAULT, metadata=IPS_METADATA)
        logger.register_metric(f'{prefix}.data_time', log.PERF_METER(),
            verbosity=dllogger.Verbosity.VERBOSE, metadata=TIME_METADATA)
        logger.register_metric(f'{prefix}.compute_latency', log.PERF_METER(
            ), verbosity=dllogger.Verbosity.VERBOSE, metadata=TIME_METADATA)
        logger.register_metric(f'{prefix}.compute_latency_at100', log.
            LAT_100(), verbosity=dllogger.Verbosity.VERBOSE, metadata=
            TIME_METADATA)
        logger.register_metric(f'{prefix}.compute_latency_at99', log.LAT_99
            (), verbosity=dllogger.Verbosity.VERBOSE, metadata=TIME_METADATA)
        logger.register_metric(f'{prefix}.compute_latency_at95', log.LAT_95
            (), verbosity=dllogger.Verbosity.VERBOSE, metadata=TIME_METADATA)
    step = get_val_step(model_and_loss, use_amp=use_amp)
    top1 = log.AverageMeter()
    model_and_loss.eval()
    end = time.time()
    data_iter = enumerate(val_loader)
    if not logger is None:
        data_iter = logger.iteration_generator_wrapper(data_iter, mode='val')
    for i, (input, target) in data_iter:
        bs = input.size(0)
        data_time = time.time() - end
        loss, prec1, prec5 = step(input, target)
        it_time = time.time() - end
        top1.record(prec1.item(), bs)
        if logger is not None:
            logger.log_metric(f'{prefix}.top1', prec1.item(), bs)
            logger.log_metric(f'{prefix}.top5', prec5.item(), bs)
            logger.log_metric(f'{prefix}.loss', loss.item(), bs)
            logger.log_metric(f'{prefix}.compute_ips', utils.calc_ips(bs, 
                it_time - data_time))
            logger.log_metric(f'{prefix}.total_ips', utils.calc_ips(bs,
                it_time))
            logger.log_metric(f'{prefix}.data_time', data_time)
            logger.log_metric(f'{prefix}.compute_latency', it_time - data_time)
            logger.log_metric(f'{prefix}.compute_latency_at95', it_time -
                data_time)
            logger.log_metric(f'{prefix}.compute_latency_at99', it_time -
                data_time)
            logger.log_metric(f'{prefix}.compute_latency_at100', it_time -
                data_time)
        end = time.time()
        if prof > 0 and i + 1 >= prof:
            time.sleep(5)
            break
    return top1.get_val()
