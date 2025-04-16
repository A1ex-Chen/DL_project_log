def test_stable_diffusion_depth2img_pipeline_k_lms(self):
    pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
        'stabilityai/stable-diffusion-2-depth', safety_checker=None)
    pipe.unet.set_default_attn_processor()
    pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    pipe.enable_attention_slicing()
    inputs = self.get_inputs()
    image = pipe(**inputs).images
    image_slice = image[0, 253:256, 253:256, -1].flatten()
    assert image.shape == (1, 480, 640, 3)
    expected_slice = np.array([0.6363, 0.6274, 0.6309, 0.637, 0.6226, 
        0.6286, 0.6213, 0.6453, 0.6306])
    assert np.abs(expected_slice - image_slice).max() < 0.0008
