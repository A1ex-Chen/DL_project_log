def test_latent_inputs(self):
    components = self.get_dummy_components()
    pipe = self.pipeline_class(**components)
    pipe.set_progress_bar_config(disable=None)
    pipe.to(torch_device)
    inputs = self.get_dummy_inputs(torch_device)
    inputs['latents'] = torch.randn((1, 4, 1, 32, 32), device=torch_device)
    inputs.pop('video')
    pipe(**inputs)
