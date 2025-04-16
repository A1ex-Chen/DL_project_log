def test_vae_tiling(self):
    components = self.get_dummy_components()
    if 'safety_checker' in components:
        components['safety_checker'] = None
    pipe = self.pipeline_class(**components)
    pipe = pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(torch_device)
    inputs['return_dict'] = False
    output_1 = pipe(**inputs)[0]
    pipe.enable_vae_tiling()
    inputs = self.get_dummy_inputs(torch_device)
    inputs['return_dict'] = False
    output_2 = pipe(**inputs)[0]
    assert np.abs(to_np(output_2) - to_np(output_1)).max() < 0.5
    shapes = [(1, 4, 73, 97), (1, 4, 97, 73), (1, 4, 49, 65), (1, 4, 65, 49)]
    with torch.no_grad():
        for shape in shapes:
            zeros = torch.zeros(shape).to(torch_device)
            pipe.vae.decode(zeros)
