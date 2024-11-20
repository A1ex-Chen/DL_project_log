def __enter__(self):
    self._interrupted = False
    self.released = False
    self.original_handler = signal.getsignal(self.sig)

    def master_handler(signum, frame):
        self.release()
        self._interrupted = True
        print(f'Received SIGTERM')

    def ignoring_handler(signum, frame):
        self.release()
        print('Received SIGTERM, ignoring')
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        signal.signal(self.sig, master_handler)
    else:
        signal.signal(self.sig, ignoring_handler)
    return self
