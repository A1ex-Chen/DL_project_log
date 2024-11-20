def test_auto_load():


    class SomeModel(Model):

        def _load(self):
            self.some_attribute = 'OK'

        def _predict(self, item):
            return self.some_attribute
    m = SomeModel()
    assert m.predict({}) == 'OK'


    class SomeModelDep(Model):

        def _load(self):
            self.some_attribute = self.model_dependencies['model'
                ].some_attribute

        def _predict(self, item):
            return self.some_attribute
    m = SomeModelDep(model_dependencies={'model': SomeModel()})
    assert m.predict({}) == 'OK'
