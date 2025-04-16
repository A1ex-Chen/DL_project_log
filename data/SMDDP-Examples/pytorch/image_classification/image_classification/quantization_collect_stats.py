def collect_stats(model, data_loader, logger, num_batches):
    """Feed data to the network and collect statistic"""
    if logger is not None:
        logger.register_metric(f'calib.total_ips', log.PERF_METER(),
            verbosity=dllogger.Verbosity.DEFAULT, metadata=IPS_METADATA)
        logger.register_metric(f'calib.data_time', log.PERF_METER(),
            verbosity=dllogger.Verbosity.DEFAULT, metadata=TIME_METADATA)
        logger.register_metric(f'calib.compute_latency', log.PERF_METER(),
            verbosity=dllogger.Verbosity.DEFAULT, metadata=TIME_METADATA)
    data_iter = enumerate(data_loader)
    if logger is not None:
        data_iter = logger.iteration_generator_wrapper(data_iter, mode='calib')
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()
    end = time.time()
    if logger is not None:
        logger.start_calibration()
    for i, (image, _) in data_iter:
        bs = image.size(0)
        data_time = time.time() - end
        model(image.cuda())
        it_time = time.time() - end
        if logger is not None:
            logger.log_metric(f'calib.total_ips', calc_ips(bs, it_time))
            logger.log_metric(f'calib.data_time', data_time)
            logger.log_metric(f'calib.compute_latency', it_time - data_time)
        if i >= num_batches:
            time.sleep(5)
            break
        end = time.time()
    if logger is not None:
        logger.end_calibration()
    logging.disable(logging.WARNING)
    disable_calibration(model)
