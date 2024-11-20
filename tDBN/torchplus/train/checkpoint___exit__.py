def __exit__(self, type, value, traceback):
    signal.signal(signal.SIGINT, self.old_handler)
    if self.signal_received:
        self.old_handler(*self.signal_received)
