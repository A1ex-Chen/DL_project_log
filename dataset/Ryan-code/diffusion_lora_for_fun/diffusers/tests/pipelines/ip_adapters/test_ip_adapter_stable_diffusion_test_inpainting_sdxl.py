def test_inpainting_sdxl(self):
    image_encoder = self.get_image_encoder(repo_id='h94/IP-Adapter',
        subfolder='sdxl_models/image_encoder')
    feature_extractor = self.get_image_processor(
        'laion/CLIP-ViT-bigG-14-laion2B-39B-b160k')
    pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(
        'stabilityai/stable-diffusion-xl-base-1.0', image_encoder=
        image_encoder, feature_extractor=feature_extractor, torch_dtype=
        self.dtype)
    pipeline.enable_model_cpu_offload()
    pipeline.load_ip_adapter('h94/IP-Adapter', subfolder='sdxl_models',
        weight_name='ip-adapter_sdxl.bin')
    inputs = self.get_dummy_inputs(for_inpainting=True)
    images = pipeline(**inputs).images
    image_slice = images[0, :3, :3, -1].flatten()
    image_slice.tolist()
    expected_slice = np.array([0.14181179, 0.1493012, 0.14283323, 
        0.14602411, 0.14915377, 0.15015268, 0.14725655, 0.15009224, 0.15164584]
        )
    max_diff = numpy_cosine_similarity_distance(image_slice, expected_slice)
    assert max_diff < 0.0005
    image_encoder = self.get_image_encoder(repo_id='h94/IP-Adapter',
        subfolder='models/image_encoder')
    feature_extractor = self.get_image_processor(
        'laion/CLIP-ViT-bigG-14-laion2B-39B-b160k')
    pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(
        'stabilityai/stable-diffusion-xl-base-1.0', image_encoder=
        image_encoder, feature_extractor=feature_extractor, torch_dtype=
        self.dtype)
    pipeline.to(torch_device)
    pipeline.load_ip_adapter('h94/IP-Adapter', subfolder='sdxl_models',
        weight_name='ip-adapter-plus_sdxl_vit-h.bin')
    inputs = self.get_dummy_inputs(for_inpainting=True)
    images = pipeline(**inputs).images
    image_slice = images[0, :3, :3, -1].flatten()
    image_slice.tolist()
    expected_slice = np.array([0.1398, 0.1476, 0.1407, 0.1442, 0.147, 0.148,
        0.1449, 0.1481, 0.1494])
    max_diff = numpy_cosine_similarity_distance(image_slice, expected_slice)
    assert max_diff < 0.0005
