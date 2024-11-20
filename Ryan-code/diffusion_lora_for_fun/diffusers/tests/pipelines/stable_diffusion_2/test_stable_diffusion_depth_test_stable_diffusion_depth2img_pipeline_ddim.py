def test_stable_diffusion_depth2img_pipeline_ddim(self):
    pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
        'stabilityai/stable-diffusion-2-depth', safety_checker=None)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    pipe.enable_attention_slicing()
    inputs = self.get_inputs()
    image = pipe(**inputs).images
    image_slice = image[0, 253:256, 253:256, -1].flatten()
    assert image.shape == (1, 480, 640, 3)
    expected_slice = np.array([0.6424, 0.6524, 0.6249, 0.6041, 0.6634, 
        0.642, 0.6522, 0.6555, 0.6436])
    assert np.abs(expected_slice - image_slice).max() < 0.0005
