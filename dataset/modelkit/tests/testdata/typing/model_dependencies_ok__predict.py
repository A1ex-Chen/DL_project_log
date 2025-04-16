def _predict(self, item):
    m = self.model_dependencies.get('dependent', SomeModel)
    res = m.predict(item)
    return res
