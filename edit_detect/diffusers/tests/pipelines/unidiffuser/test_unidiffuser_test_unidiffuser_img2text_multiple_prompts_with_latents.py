def test_unidiffuser_img2text_multiple_prompts_with_latents(self):
    device = 'cpu'
    components = self.get_dummy_components()
    unidiffuser_pipe = UniDiffuserPipeline(**components)
    unidiffuser_pipe = unidiffuser_pipe.to(device)
    unidiffuser_pipe.set_progress_bar_config(disable=None)
    unidiffuser_pipe.set_image_to_text_mode()
    assert unidiffuser_pipe.mode == 'img2text'
    inputs = self.get_dummy_inputs_with_latents(device)
    del inputs['prompt']
    inputs['num_images_per_prompt'] = 2
    inputs['num_prompts_per_image'] = 3
    text = unidiffuser_pipe(**inputs).text
    assert len(text) == 3
