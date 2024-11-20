def test_latents_input(self):
    components = self.get_dummy_components()
    pipe = StableDiffusionXLInstructPix2PixPipeline(**components)
    pipe.image_processor = VaeImageProcessor(do_resize=False, do_normalize=
        False)
    pipe = pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    out = pipe(**self.get_dummy_inputs_by_type(torch_device,
        input_image_type='pt'))[0]
    vae = components['vae']
    inputs = self.get_dummy_inputs_by_type(torch_device, input_image_type='pt')
    for image_param in self.image_latents_params:
        if image_param in inputs.keys():
            inputs[image_param] = vae.encode(inputs[image_param]
                ).latent_dist.mode()
    out_latents_inputs = pipe(**inputs)[0]
    max_diff = np.abs(out - out_latents_inputs).max()
    self.assertLess(max_diff, 0.0001,
        'passing latents as image input generate different result from passing image'
        )
