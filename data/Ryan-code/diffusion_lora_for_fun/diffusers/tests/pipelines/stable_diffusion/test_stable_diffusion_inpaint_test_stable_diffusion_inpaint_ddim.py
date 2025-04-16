def test_stable_diffusion_inpaint_ddim(self):
    vae = AsymmetricAutoencoderKL.from_pretrained(
        'cross-attention/asymmetric-autoencoder-kl-x-1-5')
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        'runwayml/stable-diffusion-inpainting', safety_checker=None)
    pipe.vae = vae
    pipe.unet.set_default_attn_processor()
    pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    pipe.enable_attention_slicing()
    inputs = self.get_inputs(torch_device)
    image = pipe(**inputs).images
    image_slice = image[0, 253:256, 253:256, -1].flatten()
    assert image.shape == (1, 512, 512, 3)
    expected_slice = np.array([0.0522, 0.0604, 0.0596, 0.0449, 0.0493, 
        0.0427, 0.1186, 0.1289, 0.1442])
    assert np.abs(expected_slice - image_slice).max() < 0.001
