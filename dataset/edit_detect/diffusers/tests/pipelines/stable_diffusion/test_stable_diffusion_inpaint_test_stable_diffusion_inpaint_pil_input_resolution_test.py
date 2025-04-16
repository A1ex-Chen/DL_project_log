def test_stable_diffusion_inpaint_pil_input_resolution_test(self):
    vae = AsymmetricAutoencoderKL.from_pretrained(
        'cross-attention/asymmetric-autoencoder-kl-x-1-5')
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        'runwayml/stable-diffusion-inpainting', safety_checker=None)
    pipe.vae = vae
    pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    pipe.enable_attention_slicing()
    inputs = self.get_inputs(torch_device)
    inputs['image'] = inputs['image'].resize((127, 127))
    inputs['mask_image'] = inputs['mask_image'].resize((127, 127))
    inputs['height'] = 128
    inputs['width'] = 128
    image = pipe(**inputs).images
    assert image.shape == (1, inputs['height'], inputs['width'], 3)
