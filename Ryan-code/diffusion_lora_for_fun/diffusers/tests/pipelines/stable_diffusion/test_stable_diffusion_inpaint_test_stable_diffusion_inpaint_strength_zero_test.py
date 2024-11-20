def test_stable_diffusion_inpaint_strength_zero_test(self):
    device = 'cpu'
    components = self.get_dummy_components()
    sd_pipe = StableDiffusionInpaintPipeline(**components)
    sd_pipe = sd_pipe.to(device)
    sd_pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(device)
    inputs['strength'] = 0.01
    with self.assertRaises(ValueError):
        sd_pipe(**inputs).images
