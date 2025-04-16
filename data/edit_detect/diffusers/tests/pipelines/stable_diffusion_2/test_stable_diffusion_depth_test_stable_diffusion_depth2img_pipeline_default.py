def test_stable_diffusion_depth2img_pipeline_default(self):
    pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
        'stabilityai/stable-diffusion-2-depth', safety_checker=None)
    pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    pipe.enable_attention_slicing()
    inputs = self.get_inputs()
    image = pipe(**inputs).images
    image_slice = image[0, 253:256, 253:256, -1].flatten()
    assert image.shape == (1, 480, 640, 3)
    expected_slice = np.array([0.5435, 0.4992, 0.3783, 0.4411, 0.5842, 
        0.4654, 0.3786, 0.5077, 0.4655])
    assert np.abs(expected_slice - image_slice).max() < 0.6
