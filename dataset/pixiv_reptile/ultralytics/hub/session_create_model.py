def create_model(self, model_args):
    """Initializes a HUB training session with the specified model identifier."""
    payload = {'config': {'batchSize': model_args.get('batch', -1),
        'epochs': model_args.get('epochs', 300), 'imageSize': model_args.
        get('imgsz', 640), 'patience': model_args.get('patience', 100),
        'device': str(model_args.get('device', '')), 'cache': str(
        model_args.get('cache', 'ram'))}, 'dataset': {'name': model_args.
        get('data')}, 'lineage': {'architecture': {'name': self.filename.
        replace('.pt', '').replace('.yaml', '')}, 'parent': {}}, 'meta': {
        'name': self.filename}}
    if self.filename.endswith('.pt'):
        payload['lineage']['parent']['name'] = self.filename
    self.model.create_model(payload)
    if not self.model.id:
        return None
    self.model_url = f'{HUB_WEB_ROOT}/models/{self.model.id}'
    self.model.start_heartbeat(self.rate_limits['heartbeat'])
    LOGGER.info(f'{PREFIX}View model at {self.model_url} 🚀')
