def test_stable_diffusion_simple_inpaint_ddim(self):
    vae = AsymmetricAutoencoderKL.from_pretrained(
        'cross-attention/asymmetric-autoencoder-kl-x-1-5')
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5', safety_checker=None)
    pipe.vae = vae
    pipe.unet.set_default_attn_processor()
    pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    pipe.enable_attention_slicing()
    inputs = self.get_inputs(torch_device)
    image = pipe(**inputs).images
    image_slice = image[0, 253:256, 253:256, -1].flatten()
    assert image.shape == (1, 512, 512, 3)
    expected_slice = np.array([0.3296, 0.4041, 0.4097, 0.4145, 0.4342, 
        0.4152, 0.4927, 0.4931, 0.443])
    assert np.abs(expected_slice - image_slice).max() < 0.001
