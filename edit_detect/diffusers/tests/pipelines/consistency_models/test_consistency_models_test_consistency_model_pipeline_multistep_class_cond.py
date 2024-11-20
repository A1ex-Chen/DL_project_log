def test_consistency_model_pipeline_multistep_class_cond(self):
    device = 'cpu'
    components = self.get_dummy_components(class_cond=True)
    pipe = ConsistencyModelPipeline(**components)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(device)
    inputs['class_labels'] = 0
    image = pipe(**inputs).images
    assert image.shape == (1, 32, 32, 3)
    image_slice = image[0, -3:, -3:, -1]
    expected_slice = np.array([0.3572, 0.6273, 0.4031, 0.3961, 0.4321, 
        0.573, 0.5266, 0.478, 0.5004])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.001
