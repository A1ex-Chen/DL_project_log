def handler(signum, frame):
    self.release()
    self.interrupted = True
    logging.info(f'Received SIGTERM')
