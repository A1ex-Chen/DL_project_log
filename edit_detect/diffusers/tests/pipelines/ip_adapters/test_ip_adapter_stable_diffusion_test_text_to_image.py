def test_text_to_image(self):
    image_encoder = self.get_image_encoder(repo_id='h94/IP-Adapter',
        subfolder='models/image_encoder')
    pipeline = StableDiffusionPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5', image_encoder=image_encoder,
        safety_checker=None, torch_dtype=self.dtype)
    pipeline.to(torch_device)
    pipeline.load_ip_adapter('h94/IP-Adapter', subfolder='models',
        weight_name='ip-adapter_sd15.bin')
    inputs = self.get_dummy_inputs()
    images = pipeline(**inputs).images
    image_slice = images[0, :3, :3, -1].flatten()
    expected_slice = np.array([0.80810547, 0.88183594, 0.9296875, 0.9189453,
        0.9848633, 1.0, 0.97021484, 1.0, 1.0])
    max_diff = numpy_cosine_similarity_distance(image_slice, expected_slice)
    assert max_diff < 0.0005
    pipeline.load_ip_adapter('h94/IP-Adapter', subfolder='models',
        weight_name='ip-adapter-plus_sd15.bin')
    inputs = self.get_dummy_inputs()
    images = pipeline(**inputs).images
    image_slice = images[0, :3, :3, -1].flatten()
    expected_slice = np.array([0.30444336, 0.26513672, 0.22436523, 
        0.2758789, 0.25585938, 0.20751953, 0.25390625, 0.24633789, 0.21923828])
    max_diff = numpy_cosine_similarity_distance(image_slice, expected_slice)
    assert max_diff < 0.0005
