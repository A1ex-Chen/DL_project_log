def test_ip_adapter_multiple_masks(self):
    image_encoder = self.get_image_encoder(repo_id='h94/IP-Adapter',
        subfolder='models/image_encoder')
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        'stabilityai/stable-diffusion-xl-base-1.0', image_encoder=
        image_encoder, torch_dtype=self.dtype)
    pipeline.enable_model_cpu_offload()
    pipeline.load_ip_adapter('h94/IP-Adapter', subfolder='sdxl_models',
        weight_name=['ip-adapter-plus-face_sdxl_vit-h.safetensors'] * 2)
    pipeline.set_ip_adapter_scale([0.7] * 2)
    inputs = self.get_dummy_inputs(for_masks=True)
    masks = inputs['cross_attention_kwargs']['ip_adapter_masks']
    processor = IPAdapterMaskProcessor()
    masks = processor.preprocess(masks)
    inputs['cross_attention_kwargs']['ip_adapter_masks'] = masks
    images = pipeline(**inputs).images
    image_slice = images[0, :3, :3, -1].flatten()
    expected_slice = np.array([0.79474676, 0.7977683, 0.8013954, 0.7988008,
        0.7970615, 0.8029355, 0.80614823, 0.8050743, 0.80627424])
    max_diff = numpy_cosine_similarity_distance(image_slice, expected_slice)
    assert max_diff < 0.0005
