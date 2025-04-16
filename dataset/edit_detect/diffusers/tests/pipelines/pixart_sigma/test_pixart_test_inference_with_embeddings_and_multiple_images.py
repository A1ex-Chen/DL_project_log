def test_inference_with_embeddings_and_multiple_images(self):
    components = self.get_dummy_components()
    pipe = self.pipeline_class(**components)
    pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(torch_device)
    prompt = inputs['prompt']
    generator = inputs['generator']
    num_inference_steps = inputs['num_inference_steps']
    output_type = inputs['output_type']
    (prompt_embeds, prompt_attn_mask, negative_prompt_embeds,
        neg_prompt_attn_mask) = pipe.encode_prompt(prompt)
    inputs = {'prompt_embeds': prompt_embeds, 'prompt_attention_mask':
        prompt_attn_mask, 'negative_prompt': None, 'negative_prompt_embeds':
        negative_prompt_embeds, 'negative_prompt_attention_mask':
        neg_prompt_attn_mask, 'generator': generator, 'num_inference_steps':
        num_inference_steps, 'output_type': output_type,
        'num_images_per_prompt': 2, 'use_resolution_binning': False}
    for optional_component in pipe._optional_components:
        setattr(pipe, optional_component, None)
    output = pipe(**inputs)[0]
    with tempfile.TemporaryDirectory() as tmpdir:
        pipe.save_pretrained(tmpdir)
        pipe_loaded = self.pipeline_class.from_pretrained(tmpdir)
        pipe_loaded.to(torch_device)
        pipe_loaded.set_progress_bar_config(disable=None)
    for optional_component in pipe._optional_components:
        self.assertTrue(getattr(pipe_loaded, optional_component) is None,
            f'`{optional_component}` did not stay set to None after loading.')
    inputs = self.get_dummy_inputs(torch_device)
    generator = inputs['generator']
    num_inference_steps = inputs['num_inference_steps']
    output_type = inputs['output_type']
    inputs = {'prompt_embeds': prompt_embeds, 'prompt_attention_mask':
        prompt_attn_mask, 'negative_prompt': None, 'negative_prompt_embeds':
        negative_prompt_embeds, 'negative_prompt_attention_mask':
        neg_prompt_attn_mask, 'generator': generator, 'num_inference_steps':
        num_inference_steps, 'output_type': output_type,
        'num_images_per_prompt': 2, 'use_resolution_binning': False}
    output_loaded = pipe_loaded(**inputs)[0]
    max_diff = np.abs(to_np(output) - to_np(output_loaded)).max()
    self.assertLess(max_diff, 0.0001)
