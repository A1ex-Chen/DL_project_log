@skip_mps
def test_image_embeds_none(self):
    device = 'cpu'
    components = self.get_dummy_components()
    sd_pipe = StableUnCLIPImg2ImgPipeline(**components)
    sd_pipe = sd_pipe.to(device)
    sd_pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(device)
    inputs.update({'image_embeds': None})
    image = sd_pipe(**inputs).images
    image_slice = image[0, -3:, -3:, -1]
    assert image.shape == (1, 32, 32, 3)
    expected_slice = np.array([0.3872, 0.7224, 0.5601, 0.4741, 0.6872, 
        0.5814, 0.4636, 0.3867, 0.5078])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.001
