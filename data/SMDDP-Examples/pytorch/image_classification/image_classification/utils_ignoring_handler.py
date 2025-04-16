def ignoring_handler(signum, frame):
    self.release()
    print('Received SIGTERM, ignoring')
