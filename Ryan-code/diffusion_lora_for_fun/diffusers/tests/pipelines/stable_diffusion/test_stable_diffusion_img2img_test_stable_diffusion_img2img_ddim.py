def test_stable_diffusion_img2img_ddim(self):
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        'CompVis/stable-diffusion-v1-4', safety_checker=None)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    pipe.enable_attention_slicing()
    inputs = self.get_inputs(torch_device)
    image = pipe(**inputs).images
    image_slice = image[0, -3:, -3:, -1].flatten()
    assert image.shape == (1, 512, 768, 3)
    expected_slice = np.array([0.0593, 0.0607, 0.0851, 0.0582, 0.0636, 
        0.0721, 0.0751, 0.0981, 0.0781])
    assert np.abs(expected_slice - image_slice).max() < 0.001
