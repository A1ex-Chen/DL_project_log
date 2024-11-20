def register_ignoring_timeout_handler(sig=signal.SIGTERM):

    def handler(signum, frame):
        logging.info('Received SIGTERM, ignoring')
    signal.signal(sig, handler)
