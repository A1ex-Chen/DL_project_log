def test_vae_slicing(self, image_count=4):
    device = 'cpu'
    components = self.get_dummy_components()
    pipe = self.pipeline_class(**components)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(device)
    inputs['prompt'] = [inputs['prompt']] * image_count
    if 'image' in inputs:
        inputs['image'] = [inputs['image']] * image_count
    output_1 = pipe(**inputs)
    pipe.enable_vae_slicing()
    inputs = self.get_dummy_inputs(device)
    inputs['prompt'] = [inputs['prompt']] * image_count
    if 'image' in inputs:
        inputs['image'] = [inputs['image']] * image_count
    inputs['return_dict'] = False
    output_2 = pipe(**inputs)
    assert np.abs(output_2[0].flatten() - output_1[0].flatten()).max() < 0.01
