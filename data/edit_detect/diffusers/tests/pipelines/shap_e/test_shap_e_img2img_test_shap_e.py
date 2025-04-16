def test_shap_e(self):
    device = 'cpu'
    components = self.get_dummy_components()
    pipe = self.pipeline_class(**components)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=None)
    output = pipe(**self.get_dummy_inputs(device))
    image = output.images[0]
    image_slice = image[-3:, -3:].cpu().numpy()
    assert image.shape == (32, 16)
    expected_slice = np.array([-1.0, 0.40668195, 0.57322013, -0.9469888, 
        0.4283227, 0.30348337, -0.81094897, 0.74555075, 0.15342723])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01
