def __init__(self, identifier):
    """
        Initialize the HUBTrainingSession with the provided model identifier.

        Args:
            identifier (str): Model identifier used to initialize the HUB training session.
                It can be a URL string or a model key with specific format.

        Raises:
            ValueError: If the provided model identifier is invalid.
            ConnectionError: If connecting with global API key is not supported.
            ModuleNotFoundError: If hub-sdk package is not installed.
        """
    from hub_sdk import HUBClient
    self.rate_limits = {'metrics': 3, 'ckpt': 900, 'heartbeat': 300}
    self.metrics_queue = {}
    self.metrics_upload_failed_queue = {}
    self.timers = {}
    self.model = None
    self.model_url = None
    api_key, model_id, self.filename = self._parse_identifier(identifier)
    active_key = api_key or SETTINGS.get('api_key')
    credentials = {'api_key': active_key} if active_key else None
    self.client = HUBClient(credentials)
    if self.client.authenticated:
        if model_id:
            self.load_model(model_id)
        else:
            self.model = self.client.model()
