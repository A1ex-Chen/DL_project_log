def test_image_to_image(self):
    image_encoder = self.get_image_encoder(repo_id='h94/IP-Adapter',
        subfolder='models/image_encoder')
    pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5', image_encoder=image_encoder,
        safety_checker=None, torch_dtype=self.dtype)
    pipeline.to(torch_device)
    pipeline.load_ip_adapter('h94/IP-Adapter', subfolder='models',
        weight_name='ip-adapter_sd15.bin')
    inputs = self.get_dummy_inputs(for_image_to_image=True)
    images = pipeline(**inputs).images
    image_slice = images[0, :3, :3, -1].flatten()
    expected_slice = np.array([0.22167969, 0.21875, 0.21728516, 0.22607422,
        0.21948242, 0.23925781, 0.22387695, 0.25268555, 0.2722168])
    max_diff = numpy_cosine_similarity_distance(image_slice, expected_slice)
    assert max_diff < 0.0005
    pipeline.load_ip_adapter('h94/IP-Adapter', subfolder='models',
        weight_name='ip-adapter-plus_sd15.bin')
    inputs = self.get_dummy_inputs(for_image_to_image=True)
    images = pipeline(**inputs).images
    image_slice = images[0, :3, :3, -1].flatten()
    expected_slice = np.array([0.35913086, 0.265625, 0.26367188, 0.24658203,
        0.19750977, 0.39990234, 0.15258789, 0.20336914, 0.5517578])
    max_diff = numpy_cosine_similarity_distance(image_slice, expected_slice)
    assert max_diff < 0.0005
