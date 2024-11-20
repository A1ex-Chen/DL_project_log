def test_unidiffuser_text2img_multiple_images(self):
    device = 'cpu'
    components = self.get_dummy_components()
    unidiffuser_pipe = UniDiffuserPipeline(**components)
    unidiffuser_pipe = unidiffuser_pipe.to(device)
    unidiffuser_pipe.set_progress_bar_config(disable=None)
    unidiffuser_pipe.set_text_to_image_mode()
    assert unidiffuser_pipe.mode == 'text2img'
    inputs = self.get_dummy_inputs(device)
    del inputs['image']
    inputs['num_images_per_prompt'] = 2
    inputs['num_prompts_per_image'] = 3
    image = unidiffuser_pipe(**inputs).images
    assert image.shape == (2, 32, 32, 3)
