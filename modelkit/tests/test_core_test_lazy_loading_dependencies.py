def test_lazy_loading_dependencies():


    class Model0(Asset):
        CONFIGURATIONS = {'model0': {}}

        def _load(self):
            self.some_attribute = 'ok'


    class Model1(Model):
        CONFIGURATIONS = {'model1': {'model_dependencies': {'model0'}}}

        def _load(self):
            self.some_attribute = self.model_dependencies['model0'
                ].some_attribute

        def _predict(self, item):
            return self.some_attribute
    p = ModelLibrary(models=[Model1, Model0], settings={'lazy_loading': True})
    m = p.get('model1')
    assert m({}) == 'ok'
    assert m.model_dependencies['model0'].some_attribute == 'ok'
    assert m.some_attribute == 'ok'
