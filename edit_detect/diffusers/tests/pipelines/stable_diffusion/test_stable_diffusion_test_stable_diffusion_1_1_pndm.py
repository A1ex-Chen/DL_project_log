def test_stable_diffusion_1_1_pndm(self):
    sd_pipe = StableDiffusionPipeline.from_pretrained(
        'CompVis/stable-diffusion-v1-1')
    sd_pipe = sd_pipe.to(torch_device)
    sd_pipe.set_progress_bar_config(disable=None)
    inputs = self.get_inputs(torch_device)
    image = sd_pipe(**inputs).images
    image_slice = image[0, -3:, -3:, -1].flatten()
    assert image.shape == (1, 512, 512, 3)
    expected_slice = np.array([0.4363, 0.4355, 0.3667, 0.4066, 0.397, 
        0.3866, 0.4394, 0.4356, 0.4059])
    assert np.abs(image_slice - expected_slice).max() < 0.003
