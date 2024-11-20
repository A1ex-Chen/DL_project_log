def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.endpoint = self.model_settings['endpoint']
    self.endpoint_headers = self.model_settings.get('endpoint_headers', {})
    self.endpoint_params = self.model_settings.get('endpoint_params', {})
    self.aiohttp_session: Optional[aiohttp.ClientSession] = None
    self.timeout = self.model_settings.get('timeout', 60)
