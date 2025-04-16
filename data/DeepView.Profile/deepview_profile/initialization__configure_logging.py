def _configure_logging(args):
    kwargs = {'format': '%(asctime)s %(levelname)-8s %(message)s',
        'datefmt': '%Y-%m-%d %H:%M', 'level': logging.DEBUG if args.debug else
        logging.INFO}
    if args.log_file is not None:
        kwargs['filename'] = args.log_file
    logging.basicConfig(**kwargs)
