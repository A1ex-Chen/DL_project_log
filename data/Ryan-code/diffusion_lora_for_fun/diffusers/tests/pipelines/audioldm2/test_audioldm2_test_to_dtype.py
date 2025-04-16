def test_to_dtype(self):
    components = self.get_dummy_components()
    pipe = self.pipeline_class(**components)
    pipe.set_progress_bar_config(disable=None)
    model_dtypes = {key: component.dtype for key, component in components.
        items() if hasattr(component, 'dtype')}
    model_dtypes.pop('text_encoder')
    self.assertTrue(all(dtype == torch.float32 for dtype in model_dtypes.
        values()))
    model_dtypes['clap_text_branch'] = components['text_encoder'
        ].text_model.dtype
    self.assertTrue(all(dtype == torch.float32 for dtype in model_dtypes.
        values()))
    pipe.to(dtype=torch.float16)
    model_dtypes = {key: component.dtype for key, component in components.
        items() if hasattr(component, 'dtype')}
    self.assertTrue(all(dtype == torch.float16 for dtype in model_dtypes.
        values()))
