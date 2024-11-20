def __enter__(self):
    self.signal_received = False
    self.old_handler = signal.signal(signal.SIGINT, self.handler)
