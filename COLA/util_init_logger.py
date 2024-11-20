def init_logger(training_args, log_level):
    logger = logging.getLogger()
    logger.setLevel(log_level)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt=
        '%Y-%m-%d %H:%M:%S')
    file_name = getattr(training_args, 'log_file', 'train.log')
    fh = logging.FileHandler(os.path.join(training_args.output_dir,
        file_name), encoding='utf-8', mode='a')
    fh.setLevel(log_level)
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_warning()
    transformers.utils.logging.disable_default_handler()
    transformers.utils.logging.enable_propagation()
    return logger
