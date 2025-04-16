def reset(self):
    self.pre_initialize(**self.pre_init_args)
    logger.debug('Initializing tracker...')
    self.tracklet = Tracklet()
    for required_type in self.required_keys:
        self.tracklet.register_observation_type(required_type)
    self.frame_count = 0
    self.update_tracklet_observations(self.init_target)
    self.post_initialize(**self.post_init_args)
    logger.debug('Tracker initialized.')
