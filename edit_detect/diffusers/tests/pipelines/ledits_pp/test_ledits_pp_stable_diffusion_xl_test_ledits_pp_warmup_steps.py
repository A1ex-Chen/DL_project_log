def test_ledits_pp_warmup_steps(self):
    device = 'cpu'
    components = self.get_dummy_components()
    pipe = LEditsPPPipelineStableDiffusionXL(**components)
    pipe = pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    inversion_inputs = self.get_dummy_inversion_inputs(device)
    inversion_inputs['image'] = inversion_inputs['image'][0]
    pipe.invert(**inversion_inputs)
    inputs = self.get_dummy_inputs(device)
    inputs['edit_warmup_steps'] = [0, 5]
    pipe(**inputs).images
    inputs['edit_warmup_steps'] = [5, 0]
    pipe(**inputs).images
    inputs['edit_warmup_steps'] = [5, 10]
    pipe(**inputs).images
    inputs['edit_warmup_steps'] = [10, 5]
    pipe(**inputs).images
