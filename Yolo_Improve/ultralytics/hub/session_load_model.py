def load_model(self, model_id):
    """Loads an existing model from Ultralytics HUB using the provided model identifier."""
    self.model = self.client.model(model_id)
    if not self.model.data:
        raise ValueError(emojis('❌ The specified HUB model does not exist'))
    self.model_url = f'{HUB_WEB_ROOT}/models/{self.model.id}'
    self._set_train_args()
    self.model.start_heartbeat(self.rate_limits['heartbeat'])
    LOGGER.info(f'{PREFIX}View model at {self.model_url} 🚀')
