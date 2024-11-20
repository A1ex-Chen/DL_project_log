def test_stable_diffusion_pix2pix_ddim(self):
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        'timbrooks/instruct-pix2pix', safety_checker=None)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    pipe.enable_attention_slicing()
    inputs = self.get_inputs()
    image = pipe(**inputs).images
    image_slice = image[0, -3:, -3:, -1].flatten()
    assert image.shape == (1, 512, 512, 3)
    expected_slice = np.array([0.3828, 0.3834, 0.3818, 0.3792, 0.3865, 
        0.3752, 0.3792, 0.3847, 0.3753])
    assert np.abs(expected_slice - image_slice).max() < 0.001
