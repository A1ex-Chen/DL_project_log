def test_instant_style_multiple_masks(self):
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        'h94/IP-Adapter', subfolder='models/image_encoder', torch_dtype=
        torch.float16).to('cuda')
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        'RunDiffusion/Juggernaut-XL-v9', torch_dtype=torch.float16,
        image_encoder=image_encoder, variant='fp16').to('cuda')
    pipeline.enable_model_cpu_offload()
    pipeline.load_ip_adapter(['ostris/ip-composition-adapter',
        'h94/IP-Adapter'], subfolder=['', 'sdxl_models'], weight_name=[
        'ip_plus_composition_sdxl.safetensors',
        'ip-adapter_sdxl_vit-h.safetensors'], image_encoder_folder=None)
    scale_1 = {'down': [[0.0, 0.0, 1.0]], 'mid': [[0.0, 0.0, 1.0]], 'up': {
        'block_0': [[0.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 0.0, 1.0]],
        'block_1': [[0.0, 0.0, 1.0]]}}
    pipeline.set_ip_adapter_scale([1.0, scale_1])
    inputs = self.get_dummy_inputs(for_instant_style=True)
    processor = IPAdapterMaskProcessor()
    masks1 = inputs['cross_attention_kwargs']['ip_adapter_masks'][0]
    masks2 = inputs['cross_attention_kwargs']['ip_adapter_masks'][1]
    masks1 = processor.preprocess(masks1, height=1024, width=1024)
    masks2 = processor.preprocess(masks2, height=1024, width=1024)
    masks2 = masks2.reshape(1, masks2.shape[0], masks2.shape[2], masks2.
        shape[3])
    inputs['cross_attention_kwargs']['ip_adapter_masks'] = [masks1, masks2]
    images = pipeline(**inputs).images
    image_slice = images[0, :3, :3, -1].flatten()
    expected_slice = np.array([0.23551631, 0.20476806, 0.14099443, 0.0, 
        0.07675594, 0.05672678, 0.0, 0.0, 0.02099729])
    max_diff = numpy_cosine_similarity_distance(image_slice, expected_slice)
    assert max_diff < 0.0005
