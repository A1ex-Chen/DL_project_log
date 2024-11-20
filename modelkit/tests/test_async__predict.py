def _predict(self, item):
    return self.model_dependencies['async_model'].predict(item)
