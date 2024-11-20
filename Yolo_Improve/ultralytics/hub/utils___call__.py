def __call__(self, cfg):
    """
        Attempts to add a new event to the events list and send events if the rate limit is reached.

        Args:
            cfg (IterableSimpleNamespace): The configuration object containing mode and task information.
        """
    if not self.enabled:
        return
    if len(self.events) < 25:
        params = {**self.metadata, 'task': cfg.task, 'model': cfg.model if 
            cfg.model in GITHUB_ASSETS_NAMES else 'custom'}
        if cfg.mode == 'export':
            params['format'] = cfg.format
        self.events.append({'name': cfg.mode, 'params': params})
    t = time.time()
    if t - self.t < self.rate_limit:
        return
    data = {'client_id': SETTINGS['uuid'], 'events': self.events}
    smart_request('post', self.url, json=data, retry=0, verbose=False)
    self.events = []
    self.t = t
