def test_consistency_model_pipeline_onestep(self):
    device = 'cpu'
    components = self.get_dummy_components()
    pipe = ConsistencyModelPipeline(**components)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(device)
    inputs['num_inference_steps'] = 1
    inputs['timesteps'] = None
    image = pipe(**inputs).images
    assert image.shape == (1, 32, 32, 3)
    image_slice = image[0, -3:, -3:, -1]
    expected_slice = np.array([0.5004, 0.5004, 0.4994, 0.5008, 0.4976, 
        0.5018, 0.499, 0.4982, 0.4987])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.001
