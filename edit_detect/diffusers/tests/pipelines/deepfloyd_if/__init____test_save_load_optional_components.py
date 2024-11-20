def _test_save_load_optional_components(self):
    components = self.get_dummy_components()
    pipe = self.pipeline_class(**components)
    pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(torch_device)
    prompt = inputs['prompt']
    generator = inputs['generator']
    num_inference_steps = inputs['num_inference_steps']
    output_type = inputs['output_type']
    if 'image' in inputs:
        image = inputs['image']
    else:
        image = None
    if 'mask_image' in inputs:
        mask_image = inputs['mask_image']
    else:
        mask_image = None
    if 'original_image' in inputs:
        original_image = inputs['original_image']
    else:
        original_image = None
    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(prompt)
    inputs = {'prompt_embeds': prompt_embeds, 'negative_prompt_embeds':
        negative_prompt_embeds, 'generator': generator,
        'num_inference_steps': num_inference_steps, 'output_type': output_type}
    if image is not None:
        inputs['image'] = image
    if mask_image is not None:
        inputs['mask_image'] = mask_image
    if original_image is not None:
        inputs['original_image'] = original_image
    for optional_component in pipe._optional_components:
        setattr(pipe, optional_component, None)
    output = pipe(**inputs)[0]
    with tempfile.TemporaryDirectory() as tmpdir:
        pipe.save_pretrained(tmpdir)
        pipe_loaded = self.pipeline_class.from_pretrained(tmpdir)
        pipe_loaded.to(torch_device)
        pipe_loaded.set_progress_bar_config(disable=None)
    pipe_loaded.unet.set_attn_processor(AttnAddedKVProcessor())
    for optional_component in pipe._optional_components:
        self.assertTrue(getattr(pipe_loaded, optional_component) is None,
            f'`{optional_component}` did not stay set to None after loading.')
    inputs = self.get_dummy_inputs(torch_device)
    generator = inputs['generator']
    num_inference_steps = inputs['num_inference_steps']
    output_type = inputs['output_type']
    inputs = {'prompt_embeds': prompt_embeds, 'negative_prompt_embeds':
        negative_prompt_embeds, 'generator': generator,
        'num_inference_steps': num_inference_steps, 'output_type': output_type}
    if image is not None:
        inputs['image'] = image
    if mask_image is not None:
        inputs['mask_image'] = mask_image
    if original_image is not None:
        inputs['original_image'] = original_image
    output_loaded = pipe_loaded(**inputs)[0]
    max_diff = np.abs(to_np(output) - to_np(output_loaded)).max()
    self.assertLess(max_diff, 0.0001)
