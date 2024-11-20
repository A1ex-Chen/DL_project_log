def test_stable_diffusion_pix2pix_default(self):
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        'timbrooks/instruct-pix2pix', safety_checker=None)
    pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    pipe.enable_attention_slicing()
    inputs = self.get_inputs()
    image = pipe(**inputs).images
    image_slice = image[0, -3:, -3:, -1].flatten()
    assert image.shape == (1, 512, 512, 3)
    expected_slice = np.array([0.5902, 0.6015, 0.6027, 0.5983, 0.6092, 
        0.6061, 0.5765, 0.5785, 0.5555])
    assert np.abs(expected_slice - image_slice).max() < 0.001
