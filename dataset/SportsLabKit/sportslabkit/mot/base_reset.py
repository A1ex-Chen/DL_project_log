def reset(self):
    logger.debug('Initializing tracker...')
    self.alive_tracklets = []
    self.dead_tracklets = []
    self.frame_count = 0
    logger.debug('Tracker initialized.')
