def test_stable_diffusion_img2img_k_lms(self):
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        'CompVis/stable-diffusion-v1-4', safety_checker=None)
    pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    pipe.enable_attention_slicing()
    inputs = self.get_inputs(torch_device)
    image = pipe(**inputs).images
    image_slice = image[0, -3:, -3:, -1].flatten()
    assert image.shape == (1, 512, 768, 3)
    expected_slice = np.array([0.0389, 0.0346, 0.0415, 0.029, 0.0218, 0.021,
        0.0408, 0.0567, 0.0271])
    assert np.abs(expected_slice - image_slice).max() < 0.001
