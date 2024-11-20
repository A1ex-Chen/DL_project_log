@is_flaky
def test_multi(self):
    image_encoder = self.get_image_encoder(repo_id='h94/IP-Adapter',
        subfolder='models/image_encoder')
    pipeline = StableDiffusionPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5', image_encoder=image_encoder,
        safety_checker=None, torch_dtype=self.dtype)
    pipeline.to(torch_device)
    pipeline.load_ip_adapter('h94/IP-Adapter', subfolder='models',
        weight_name=['ip-adapter_sd15.bin', 'ip-adapter-plus_sd15.bin'])
    pipeline.set_ip_adapter_scale([0.7, 0.3])
    inputs = self.get_dummy_inputs()
    ip_adapter_image = inputs['ip_adapter_image']
    inputs['ip_adapter_image'] = [ip_adapter_image, [ip_adapter_image] * 2]
    images = pipeline(**inputs).images
    image_slice = images[0, :3, :3, -1].flatten()
    expected_slice = np.array([0.5234, 0.5352, 0.5625, 0.5713, 0.5947, 
        0.6206, 0.5786, 0.6187, 0.6494])
    max_diff = numpy_cosine_similarity_distance(image_slice, expected_slice)
    assert max_diff < 0.0005
