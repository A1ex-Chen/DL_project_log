def _test_save_load_optional_components(self, expected_max_difference=0.0001):
    components = self.get_dummy_components()
    pipe = self.pipeline_class(**components)
    for optional_component in pipe._optional_components:
        setattr(pipe, optional_component, None)
    for component in pipe.components.values():
        if hasattr(component, 'set_default_attn_processor'):
            component.set_default_attn_processor()
    pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    generator_device = 'cpu'
    inputs = self.get_dummy_inputs(generator_device)
    tokenizer = components.pop('tokenizer')
    tokenizer_2 = components.pop('tokenizer_2')
    text_encoder = components.pop('text_encoder')
    text_encoder_2 = components.pop('text_encoder_2')
    tokenizers = [tokenizer, tokenizer_2] if tokenizer is not None else [
        tokenizer_2]
    text_encoders = [text_encoder, text_encoder_2
        ] if text_encoder is not None else [text_encoder_2]
    prompt = inputs.pop('prompt')
    (prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds,
        negative_pooled_prompt_embeds) = self.encode_prompt(tokenizers,
        text_encoders, prompt)
    inputs['prompt_embeds'] = prompt_embeds
    inputs['negative_prompt_embeds'] = negative_prompt_embeds
    inputs['pooled_prompt_embeds'] = pooled_prompt_embeds
    inputs['negative_pooled_prompt_embeds'] = negative_pooled_prompt_embeds
    output = pipe(**inputs)[0]
    with tempfile.TemporaryDirectory() as tmpdir:
        pipe.save_pretrained(tmpdir)
        pipe_loaded = self.pipeline_class.from_pretrained(tmpdir)
        for component in pipe_loaded.components.values():
            if hasattr(component, 'set_default_attn_processor'):
                component.set_default_attn_processor()
        pipe_loaded.to(torch_device)
        pipe_loaded.set_progress_bar_config(disable=None)
    for optional_component in pipe._optional_components:
        self.assertTrue(getattr(pipe_loaded, optional_component) is None,
            f'`{optional_component}` did not stay set to None after loading.')
    inputs = self.get_dummy_inputs(generator_device)
    _ = inputs.pop('prompt')
    inputs['prompt_embeds'] = prompt_embeds
    inputs['negative_prompt_embeds'] = negative_prompt_embeds
    inputs['pooled_prompt_embeds'] = pooled_prompt_embeds
    inputs['negative_pooled_prompt_embeds'] = negative_pooled_prompt_embeds
    output_loaded = pipe_loaded(**inputs)[0]
    max_diff = np.abs(to_np(output) - to_np(output_loaded)).max()
    self.assertLess(max_diff, expected_max_difference)
