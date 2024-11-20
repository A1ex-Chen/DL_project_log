def test_unidiffuser_default_joint_v1(self):
    pipe = UniDiffuserPipeline.from_pretrained('thu-ml/unidiffuser-v1')
    pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    pipe.enable_attention_slicing()
    inputs = self.get_inputs(device=torch_device, generate_latents=True)
    del inputs['prompt']
    del inputs['image']
    sample = pipe(**inputs)
    image = sample.images
    text = sample.text
    assert image.shape == (1, 512, 512, 3)
    image_slice = image[0, -3:, -3:, -1]
    expected_img_slice = np.array([0.2402, 0.2375, 0.2285, 0.2378, 0.2407, 
        0.2263, 0.2354, 0.2307, 0.252])
    assert np.abs(image_slice.flatten() - expected_img_slice).max() < 0.1
    expected_text_prefix = 'a living room'
    assert text[0][:len(expected_text_prefix)] == expected_text_prefix
