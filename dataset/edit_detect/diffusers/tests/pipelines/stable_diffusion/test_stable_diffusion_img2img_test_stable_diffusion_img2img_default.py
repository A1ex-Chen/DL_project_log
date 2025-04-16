def test_stable_diffusion_img2img_default(self):
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        'CompVis/stable-diffusion-v1-4', safety_checker=None)
    pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    pipe.enable_attention_slicing()
    inputs = self.get_inputs(torch_device)
    image = pipe(**inputs).images
    image_slice = image[0, -3:, -3:, -1].flatten()
    assert image.shape == (1, 512, 768, 3)
    expected_slice = np.array([0.43, 0.4662, 0.493, 0.399, 0.4307, 0.4525, 
        0.3719, 0.4064, 0.3923])
    assert np.abs(expected_slice - image_slice).max() < 0.001
