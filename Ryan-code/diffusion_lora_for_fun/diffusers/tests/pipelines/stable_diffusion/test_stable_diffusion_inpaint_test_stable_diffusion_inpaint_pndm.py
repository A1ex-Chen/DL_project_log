def test_stable_diffusion_inpaint_pndm(self):
    vae = AsymmetricAutoencoderKL.from_pretrained(
        'cross-attention/asymmetric-autoencoder-kl-x-1-5')
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        'runwayml/stable-diffusion-inpainting', safety_checker=None)
    pipe.unet.set_default_attn_processor()
    pipe.vae = vae
    pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)
    pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    pipe.enable_attention_slicing()
    inputs = self.get_inputs(torch_device)
    image = pipe(**inputs).images
    image_slice = image[0, 253:256, 253:256, -1].flatten()
    assert image.shape == (1, 512, 512, 3)
    expected_slice = np.array([0.0966, 0.1083, 0.1148, 0.1422, 0.1318, 
        0.1197, 0.3702, 0.3537, 0.3288])
    assert np.abs(expected_slice - image_slice).max() < 0.005
