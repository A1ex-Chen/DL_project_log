def test_stable_diffusion_inpaint_strength_test(self):
    vae = AsymmetricAutoencoderKL.from_pretrained(
        'cross-attention/asymmetric-autoencoder-kl-x-1-5')
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        'runwayml/stable-diffusion-inpainting', safety_checker=None)
    pipe.unet.set_default_attn_processor()
    pipe.vae = vae
    pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    pipe.enable_attention_slicing()
    inputs = self.get_inputs(torch_device)
    inputs['strength'] = 0.75
    image = pipe(**inputs).images
    assert image.shape == (1, 512, 512, 3)
    image_slice = image[0, 253:256, 253:256, -1].flatten()
    expected_slice = np.array([0.2458, 0.2576, 0.3124, 0.2679, 0.2669, 
        0.2796, 0.2872, 0.2975, 0.2661])
    assert np.abs(expected_slice - image_slice).max() < 0.003
