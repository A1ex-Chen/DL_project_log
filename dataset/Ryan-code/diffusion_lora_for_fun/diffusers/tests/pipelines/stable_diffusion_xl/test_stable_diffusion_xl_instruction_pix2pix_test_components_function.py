def test_components_function(self):
    init_components = self.get_dummy_components()
    pipe = self.pipeline_class(**init_components)
    self.assertTrue(hasattr(pipe, 'components'))
    self.assertTrue(set(pipe.components.keys()) == set(init_components.keys()))
