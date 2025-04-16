def test_stable_diffusion_height_width_opt(self):
    components = self.get_dummy_components()
    components['scheduler'] = LMSDiscreteScheduler.from_config(components[
        'scheduler'].config)
    sd_pipe = StableDiffusionPipeline(**components)
    sd_pipe = sd_pipe.to(torch_device)
    sd_pipe.set_progress_bar_config(disable=None)
    prompt = 'hey'
    output = sd_pipe(prompt, num_inference_steps=1, output_type='np')
    image_shape = output.images[0].shape[:2]
    assert image_shape == (64, 64)
    output = sd_pipe(prompt, num_inference_steps=1, height=96, width=96,
        output_type='np')
    image_shape = output.images[0].shape[:2]
    assert image_shape == (96, 96)
    config = dict(sd_pipe.unet.config)
    config['sample_size'] = 96
    sd_pipe.unet = UNet2DConditionModel.from_config(config).to(torch_device)
    output = sd_pipe(prompt, num_inference_steps=1, output_type='np')
    image_shape = output.images[0].shape[:2]
    assert image_shape == (192, 192)
