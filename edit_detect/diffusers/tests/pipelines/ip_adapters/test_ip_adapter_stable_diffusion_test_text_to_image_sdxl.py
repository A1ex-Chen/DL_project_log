def test_text_to_image_sdxl(self):
    image_encoder = self.get_image_encoder(repo_id='h94/IP-Adapter',
        subfolder='sdxl_models/image_encoder')
    feature_extractor = self.get_image_processor(
        'laion/CLIP-ViT-bigG-14-laion2B-39B-b160k')
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        'stabilityai/stable-diffusion-xl-base-1.0', image_encoder=
        image_encoder, feature_extractor=feature_extractor, torch_dtype=
        self.dtype)
    pipeline.enable_model_cpu_offload()
    pipeline.load_ip_adapter('h94/IP-Adapter', subfolder='sdxl_models',
        weight_name='ip-adapter_sdxl.bin')
    inputs = self.get_dummy_inputs()
    images = pipeline(**inputs).images
    image_slice = images[0, :3, :3, -1].flatten()
    expected_slice = np.array([0.09630299, 0.09551358, 0.08480701, 
        0.09070173, 0.09437338, 0.09264627, 0.08883232, 0.09287417, 0.09197289]
        )
    max_diff = numpy_cosine_similarity_distance(image_slice, expected_slice)
    assert max_diff < 0.0005
    image_encoder = self.get_image_encoder(repo_id='h94/IP-Adapter',
        subfolder='models/image_encoder')
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        'stabilityai/stable-diffusion-xl-base-1.0', image_encoder=
        image_encoder, feature_extractor=feature_extractor, torch_dtype=
        self.dtype)
    pipeline.to(torch_device)
    pipeline.load_ip_adapter('h94/IP-Adapter', subfolder='sdxl_models',
        weight_name='ip-adapter-plus_sdxl_vit-h.bin')
    inputs = self.get_dummy_inputs()
    images = pipeline(**inputs).images
    image_slice = images[0, :3, :3, -1].flatten()
    expected_slice = np.array([0.0576596, 0.05600825, 0.04479006, 
        0.05288461, 0.05461192, 0.05137569, 0.04867965, 0.05301541, 0.04939842]
        )
    max_diff = numpy_cosine_similarity_distance(image_slice, expected_slice)
    assert max_diff < 0.0005
