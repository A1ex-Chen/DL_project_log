def test_stable_diffusion_depth2img_pil(self):
    device = 'cpu'
    components = self.get_dummy_components()
    pipe = StableDiffusionDepth2ImgPipeline(**components)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(device)
    image = pipe(**inputs).images
    image_slice = image[0, -3:, -3:, -1]
    if torch_device == 'mps':
        expected_slice = np.array([0.53232, 0.47015, 0.40868, 0.45651, 
            0.4891, 0.4668, 0.4287, 0.48822, 0.47439])
    else:
        expected_slice = np.array([0.5435, 0.4992, 0.3783, 0.4411, 0.5842, 
            0.4654, 0.3786, 0.5077, 0.4655])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.001
