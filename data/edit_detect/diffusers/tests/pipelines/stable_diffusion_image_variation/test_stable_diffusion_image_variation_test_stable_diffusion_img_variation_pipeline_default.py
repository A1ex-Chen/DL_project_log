def test_stable_diffusion_img_variation_pipeline_default(self):
    sd_pipe = StableDiffusionImageVariationPipeline.from_pretrained(
        'lambdalabs/sd-image-variations-diffusers', safety_checker=None)
    sd_pipe = sd_pipe.to(torch_device)
    sd_pipe.set_progress_bar_config(disable=None)
    generator_device = 'cpu'
    inputs = self.get_inputs(generator_device)
    image = sd_pipe(**inputs).images
    image_slice = image[0, -3:, -3:, -1].flatten()
    assert image.shape == (1, 512, 512, 3)
    expected_slice = np.array([0.8449, 0.9079, 0.7571, 0.7873, 0.8348, 
        0.701, 0.6694, 0.6873, 0.6138])
    max_diff = numpy_cosine_similarity_distance(image_slice, expected_slice)
    assert max_diff < 0.0001
