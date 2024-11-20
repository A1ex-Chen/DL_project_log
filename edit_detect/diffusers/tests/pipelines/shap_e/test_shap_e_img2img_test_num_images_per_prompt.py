def test_num_images_per_prompt(self):
    components = self.get_dummy_components()
    pipe = self.pipeline_class(**components)
    pipe = pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    batch_size = 1
    num_images_per_prompt = 2
    inputs = self.get_dummy_inputs(torch_device)
    for key in inputs.keys():
        if key in self.batch_params:
            inputs[key] = batch_size * [inputs[key]]
    images = pipe(**inputs, num_images_per_prompt=num_images_per_prompt)[0]
    assert images.shape[0] == batch_size * num_images_per_prompt
