def test_pipeline_signature(self):
    parameters = inspect.signature(self.pipeline_class.__call__).parameters
    assert issubclass(self.pipeline_class, IPAdapterMixin)
    self.assertIn('ip_adapter_image', parameters,
        '`ip_adapter_image` argument must be supported by the `__call__` method'
        )
    self.assertIn('ip_adapter_image_embeds', parameters,
        '`ip_adapter_image_embeds` argument must be supported by the `__call__` method'
        )
