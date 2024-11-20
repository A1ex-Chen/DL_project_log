def handler(self, sig, frame):
    self.signal_received = sig, frame
    logging.debug('SIGINT received. Delaying KeyboardInterrupt.')
