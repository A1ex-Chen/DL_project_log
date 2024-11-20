def preload(self):
    _ = self.assets_manager
    for model_name in self.required_models:
        self._load(model_name)
