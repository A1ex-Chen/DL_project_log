def __enter__(self):
    self.interrupted = False
    self.released = False
    self.original_handler = signal.getsignal(self.sig)

    def handler(signum, frame):
        self.release()
        self.interrupted = True
        logging.info(f'Received SIGTERM')
    signal.signal(self.sig, handler)
    return self
