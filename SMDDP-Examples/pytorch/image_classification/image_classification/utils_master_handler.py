def master_handler(signum, frame):
    self.release()
    self._interrupted = True
    print(f'Received SIGTERM')
