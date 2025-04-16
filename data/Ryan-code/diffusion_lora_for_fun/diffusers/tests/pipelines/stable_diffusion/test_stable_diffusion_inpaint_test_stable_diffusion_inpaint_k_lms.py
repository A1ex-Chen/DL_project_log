def test_stable_diffusion_inpaint_k_lms(self):
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
    image = pipe(**inputs).images
    image_slice = image[0, 253:256, 253:256, -1].flatten()
    assert image.shape == (1, 512, 512, 3)
    expected_slice = np.array([0.8931, 0.8683, 0.8965, 0.8501, 0.8592, 
        0.9118, 0.8734, 0.7463, 0.899])
    assert np.abs(expected_slice - image_slice).max() < 0.006
