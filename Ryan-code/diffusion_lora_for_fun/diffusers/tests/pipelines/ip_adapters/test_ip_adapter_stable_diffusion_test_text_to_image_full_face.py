def test_text_to_image_full_face(self):
    image_encoder = self.get_image_encoder(repo_id='h94/IP-Adapter',
        subfolder='models/image_encoder')
    pipeline = StableDiffusionPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5', image_encoder=image_encoder,
        safety_checker=None, torch_dtype=self.dtype)
    pipeline.to(torch_device)
    pipeline.load_ip_adapter('h94/IP-Adapter', subfolder='models',
        weight_name='ip-adapter-full-face_sd15.bin')
    pipeline.set_ip_adapter_scale(0.7)
    inputs = self.get_dummy_inputs()
    images = pipeline(**inputs).images
    image_slice = images[0, :3, :3, -1].flatten()
    expected_slice = np.array([0.1704, 0.1296, 0.1272, 0.2212, 0.1514, 
        0.1479, 0.4172, 0.4263, 0.436])
    max_diff = numpy_cosine_similarity_distance(image_slice, expected_slice)
    assert max_diff < 0.0005
