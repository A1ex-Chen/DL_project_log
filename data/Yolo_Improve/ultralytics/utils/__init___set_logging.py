def set_logging(name='LOGGING_NAME', verbose=True):
    """Sets up logging for the given name with UTF-8 encoding support, ensuring compatibility across different
    environments.
    """
    level = logging.INFO if verbose and RANK in {-1, 0} else logging.ERROR
    formatter = logging.Formatter('%(message)s')
    if WINDOWS and hasattr(sys.stdout, 'encoding'
        ) and sys.stdout.encoding != 'utf-8':


        class CustomFormatter(logging.Formatter):

            def format(self, record):
                """Sets up logging with UTF-8 encoding and configurable verbosity."""
                return emojis(super().format(record))
        try:
            if hasattr(sys.stdout, 'reconfigure'):
                sys.stdout.reconfigure(encoding='utf-8')
            elif hasattr(sys.stdout, 'buffer'):
                import io
                sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding=
                    'utf-8')
            else:
                formatter = CustomFormatter('%(message)s')
        except Exception as e:
            print(
                f'Creating custom formatter for non UTF-8 environments due to {e}'
                )
            formatter = CustomFormatter('%(message)s')
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger
