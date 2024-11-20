def test_stable_diffusion_panorama_k_lms(self):
    pipe = StableDiffusionPanoramaPipeline.from_pretrained(
        'stabilityai/stable-diffusion-2-base', safety_checker=None)
    pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.unet.set_default_attn_processor()
    pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    pipe.enable_attention_slicing()
    inputs = self.get_inputs()
    image = pipe(**inputs).images
    image_slice = image[0, -3:, -3:, -1].flatten()
    assert image.shape == (1, 512, 2048, 3)
    expected_slice = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    assert np.abs(expected_slice - image_slice).max() < 0.01
