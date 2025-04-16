def test_to_dtype(self):
    components = self.get_dummy_components()
    pipe = self.pipeline_class(**components)
    pipe.set_progress_bar_config(disable=None)
    model_dtypes = [component.dtype for component in components.values() if
        hasattr(component, 'dtype')]
    self.assertTrue(all(dtype == torch.float32 for dtype in model_dtypes))
    pipe.to(dtype=torch.float16)
    model_dtypes = [component.dtype for component in components.values() if
        hasattr(component, 'dtype')]
    self.assertTrue(all(dtype == torch.float16 for dtype in model_dtypes))
