def test_inpainting(self):
    image_encoder = self.get_image_encoder(repo_id='h94/IP-Adapter',
        subfolder='models/image_encoder')
    pipeline = StableDiffusionInpaintPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5', image_encoder=image_encoder,
        safety_checker=None, torch_dtype=self.dtype)
    pipeline.to(torch_device)
    pipeline.load_ip_adapter('h94/IP-Adapter', subfolder='models',
        weight_name='ip-adapter_sd15.bin')
    inputs = self.get_dummy_inputs(for_inpainting=True)
    images = pipeline(**inputs).images
    image_slice = images[0, :3, :3, -1].flatten()
    expected_slice = np.array([0.27148438, 0.24047852, 0.22167969, 
        0.23217773, 0.21118164, 0.21142578, 0.21875, 0.20751953, 0.20019531])
    max_diff = numpy_cosine_similarity_distance(image_slice, expected_slice)
    assert max_diff < 0.0005
    pipeline.load_ip_adapter('h94/IP-Adapter', subfolder='models',
        weight_name='ip-adapter-plus_sd15.bin')
    inputs = self.get_dummy_inputs(for_inpainting=True)
    images = pipeline(**inputs).images
    image_slice = images[0, :3, :3, -1].flatten()
    max_diff = numpy_cosine_similarity_distance(image_slice, expected_slice)
    assert max_diff < 0.0005
