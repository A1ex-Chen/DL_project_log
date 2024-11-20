def test_stable_diffusion_pix2pix_k_lms(self):
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        'timbrooks/instruct-pix2pix', safety_checker=None)
    pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    pipe.enable_attention_slicing()
    inputs = self.get_inputs()
    image = pipe(**inputs).images
    image_slice = image[0, -3:, -3:, -1].flatten()
    assert image.shape == (1, 512, 512, 3)
    expected_slice = np.array([0.6578, 0.6817, 0.6972, 0.6761, 0.6856, 
        0.6916, 0.6428, 0.6516, 0.6301])
    assert np.abs(expected_slice - image_slice).max() < 0.001
