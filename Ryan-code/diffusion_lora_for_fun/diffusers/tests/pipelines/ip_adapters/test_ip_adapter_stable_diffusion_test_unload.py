def test_unload(self):
    image_encoder = self.get_image_encoder(repo_id='h94/IP-Adapter',
        subfolder='models/image_encoder')
    pipeline = StableDiffusionPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5', image_encoder=image_encoder,
        safety_checker=None, torch_dtype=self.dtype)
    before_processors = [attn_proc.__class__ for attn_proc in pipeline.unet
        .attn_processors.values()]
    pipeline.to(torch_device)
    pipeline.load_ip_adapter('h94/IP-Adapter', subfolder='models',
        weight_name='ip-adapter_sd15.bin')
    pipeline.set_ip_adapter_scale(0.7)
    pipeline.unload_ip_adapter()
    assert getattr(pipeline, 'image_encoder') is None
    assert getattr(pipeline, 'feature_extractor') is not None
    after_processors = [attn_proc.__class__ for attn_proc in pipeline.unet.
        attn_processors.values()]
    assert before_processors == after_processors
