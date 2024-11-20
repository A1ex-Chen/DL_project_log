def test_prompt_embeds(self):
    components = self.get_dummy_components()
    pipe = self.pipeline_class(**components)
    pipe.set_progress_bar_config(disable=None)
    pipe.to(torch_device)
    inputs = self.get_dummy_inputs(torch_device)
    inputs.pop('prompt')
    inputs['prompt_embeds'] = torch.randn((1, 4, 32), device=torch_device)
    pipe(**inputs)
