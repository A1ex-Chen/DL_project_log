def test_model_dependencies_bad_get():


    class SomeModel(Model):
        CONFIGURATIONS = {'some_model': {}}

        def _load(self):
            self.some_attribute = 'OK'

        def _predict(self, item):
            return self.some_attribute


    class SomeModelDep(Model):
        CONFIGURATIONS = {'some_model_dep': {'model_dependencies': {
            'some_model'}}}

        def _load(self):
            dependencies = [x for x in self.model_dependencies]
            assert dependencies == ['some_model']
            assert len([x for x in self.model_dependencies.values()])
            assert len([x for x in self.model_dependencies.items()])
            assert len([x for x in self.model_dependencies.keys()])
            assert len(self.model_dependencies) == 1
            self.some_attribute = self.model_dependencies.get('some_model',
                SomeModel).some_attribute
            with pytest.raises(ValueError):
                _ = self.model_dependencies.get('some_model', SomeModelDep
                    ).some_attribute

        def _predict(self, item):
            return item
    lib = ModelLibrary(models=[SomeModel, SomeModelDep], required_models=[
        'some_model_dep'])
    lib.get('some_model_dep')
