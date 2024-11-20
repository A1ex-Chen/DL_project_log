def test_stable_diffusion_pix2pix_euler(self):
    device = 'cpu'
    components = self.get_dummy_components()
    components['scheduler'] = EulerAncestralDiscreteScheduler(beta_start=
        0.00085, beta_end=0.012, beta_schedule='scaled_linear')
    sd_pipe = StableDiffusionInstructPix2PixPipeline(**components)
    sd_pipe = sd_pipe.to(device)
    sd_pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(device)
    image = sd_pipe(**inputs).images
    image_slice = image[0, -3:, -3:, -1]
    slice = [round(x, 4) for x in image_slice.flatten().tolist()]
    print(','.join([str(x) for x in slice]))
    assert image.shape == (1, 32, 32, 3)
    expected_slice = np.array([0.7417, 0.3842, 0.4732, 0.5776, 0.5891, 
        0.5139, 0.4052, 0.5673, 0.4986])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.001
