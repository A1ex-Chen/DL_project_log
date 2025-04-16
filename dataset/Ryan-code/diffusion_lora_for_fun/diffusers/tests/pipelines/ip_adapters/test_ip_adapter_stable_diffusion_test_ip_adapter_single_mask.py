def test_ip_adapter_single_mask(self):
    image_encoder = self.get_image_encoder(repo_id='h94/IP-Adapter',
        subfolder='models/image_encoder')
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        'stabilityai/stable-diffusion-xl-base-1.0', image_encoder=
        image_encoder, torch_dtype=self.dtype)
    pipeline.enable_model_cpu_offload()
    pipeline.load_ip_adapter('h94/IP-Adapter', subfolder='sdxl_models',
        weight_name='ip-adapter-plus-face_sdxl_vit-h.safetensors')
    pipeline.set_ip_adapter_scale(0.7)
    inputs = self.get_dummy_inputs(for_masks=True)
    mask = inputs['cross_attention_kwargs']['ip_adapter_masks'][0]
    processor = IPAdapterMaskProcessor()
    mask = processor.preprocess(mask)
    inputs['cross_attention_kwargs']['ip_adapter_masks'] = mask
    inputs['ip_adapter_image'] = inputs['ip_adapter_image'][0]
    images = pipeline(**inputs).images
    image_slice = images[0, :3, :3, -1].flatten()
    expected_slice = np.array([0.7307304, 0.73450166, 0.73731124, 0.7377061,
        0.7318013, 0.73720926, 0.74746597, 0.7409929, 0.74074936])
    max_diff = numpy_cosine_similarity_distance(image_slice, expected_slice)
    assert max_diff < 0.0005
