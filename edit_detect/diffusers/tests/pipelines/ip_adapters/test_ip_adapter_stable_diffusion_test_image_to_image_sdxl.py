def test_image_to_image_sdxl(self):
    image_encoder = self.get_image_encoder(repo_id='h94/IP-Adapter',
        subfolder='sdxl_models/image_encoder')
    feature_extractor = self.get_image_processor(
        'laion/CLIP-ViT-bigG-14-laion2B-39B-b160k')
    pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        'stabilityai/stable-diffusion-xl-base-1.0', image_encoder=
        image_encoder, feature_extractor=feature_extractor, torch_dtype=
        self.dtype)
    pipeline.enable_model_cpu_offload()
    pipeline.load_ip_adapter('h94/IP-Adapter', subfolder='sdxl_models',
        weight_name='ip-adapter_sdxl.bin')
    inputs = self.get_dummy_inputs(for_image_to_image=True)
    images = pipeline(**inputs).images
    image_slice = images[0, :3, :3, -1].flatten()
    expected_slice = np.array([0.06513795, 0.07009393, 0.07234055, 
        0.07426041, 0.07002589, 0.06415862, 0.07827643, 0.07962808, 0.07411247]
        )
    assert np.allclose(image_slice, expected_slice, atol=0.001)
    image_encoder = self.get_image_encoder(repo_id='h94/IP-Adapter',
        subfolder='models/image_encoder')
    feature_extractor = self.get_image_processor(
        'laion/CLIP-ViT-bigG-14-laion2B-39B-b160k')
    pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        'stabilityai/stable-diffusion-xl-base-1.0', image_encoder=
        image_encoder, feature_extractor=feature_extractor, torch_dtype=
        self.dtype)
    pipeline.to(torch_device)
    pipeline.load_ip_adapter('h94/IP-Adapter', subfolder='sdxl_models',
        weight_name='ip-adapter-plus_sdxl_vit-h.bin')
    inputs = self.get_dummy_inputs(for_image_to_image=True)
    images = pipeline(**inputs).images
    image_slice = images[0, :3, :3, -1].flatten()
    expected_slice = np.array([0.07126552, 0.07025367, 0.07348302, 
        0.07580167, 0.07467338, 0.06918576, 0.07480252, 0.08279955, 0.08547315]
        )
    assert np.allclose(image_slice, expected_slice, atol=0.001)
