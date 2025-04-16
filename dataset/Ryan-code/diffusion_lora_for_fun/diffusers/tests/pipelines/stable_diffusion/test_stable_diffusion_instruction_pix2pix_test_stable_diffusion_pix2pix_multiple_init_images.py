def test_stable_diffusion_pix2pix_multiple_init_images(self):
    device = 'cpu'
    components = self.get_dummy_components()
    sd_pipe = StableDiffusionInstructPix2PixPipeline(**components)
    sd_pipe = sd_pipe.to(device)
    sd_pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(device)
    inputs['prompt'] = [inputs['prompt']] * 2
    image = np.array(inputs['image']).astype(np.float32) / 255.0
    image = torch.from_numpy(image).unsqueeze(0).to(device)
    image = image / 2 + 0.5
    image = image.permute(0, 3, 1, 2)
    inputs['image'] = image.repeat(2, 1, 1, 1)
    image = sd_pipe(**inputs).images
    image_slice = image[-1, -3:, -3:, -1]
    assert image.shape == (2, 32, 32, 3)
    expected_slice = np.array([0.5812, 0.5748, 0.5222, 0.5908, 0.5695, 
        0.7174, 0.6804, 0.5523, 0.5579])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.001
