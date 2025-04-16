def test_unidiffuser_default_text2img_v1(self):
    pipe = UniDiffuserPipeline.from_pretrained('thu-ml/unidiffuser-v1')
    pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    pipe.enable_attention_slicing()
    inputs = self.get_inputs(device=torch_device, generate_latents=True)
    del inputs['image']
    sample = pipe(**inputs)
    image = sample.images
    assert image.shape == (1, 512, 512, 3)
    image_slice = image[0, -3:, -3:, -1]
    expected_slice = np.array([0.0242, 0.0103, 0.0022, 0.0129, 0.0, 0.009, 
        0.0376, 0.0508, 0.0005])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.1
