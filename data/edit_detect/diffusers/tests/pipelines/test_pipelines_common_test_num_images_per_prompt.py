def test_num_images_per_prompt(self):
    sig = inspect.signature(self.pipeline_class.__call__)
    if 'num_images_per_prompt' not in sig.parameters:
        return
    components = self.get_dummy_components()
    pipe = self.pipeline_class(**components)
    pipe = pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    batch_sizes = [1, 2]
    num_images_per_prompts = [1, 2]
    for batch_size in batch_sizes:
        for num_images_per_prompt in num_images_per_prompts:
            inputs = self.get_dummy_inputs(torch_device)
            for key in inputs.keys():
                if key in self.batch_params:
                    inputs[key] = batch_size * [inputs[key]]
            images = pipe(**inputs, num_images_per_prompt=num_images_per_prompt
                )[0]
            assert images.shape[0] == batch_size * num_images_per_prompt
