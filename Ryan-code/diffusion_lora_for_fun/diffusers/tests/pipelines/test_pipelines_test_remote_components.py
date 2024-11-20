def test_remote_components(self):
    with self.assertRaises(ValueError):
        pipeline = DiffusionPipeline.from_pretrained(
            'hf-internal-testing/tiny-sdxl-custom-components')
    pipeline = DiffusionPipeline.from_pretrained(
        'hf-internal-testing/tiny-sdxl-custom-components',
        trust_remote_code=True)
    assert pipeline.config.unet == ('diffusers_modules.local.my_unet_model',
        'MyUNetModel')
    assert pipeline.config.scheduler == ('diffusers_modules.local.my_scheduler'
        , 'MyScheduler')
    assert pipeline.__class__.__name__ == 'StableDiffusionXLPipeline'
    pipeline = pipeline.to(torch_device)
    images = pipeline('test', num_inference_steps=2, output_type='np')[0]
    assert images.shape == (1, 64, 64, 3)
    pipeline = DiffusionPipeline.from_pretrained(
        'hf-internal-testing/tiny-sdxl-custom-components', custom_pipeline=
        'my_pipeline', trust_remote_code=True)
    assert pipeline.config.unet == ('diffusers_modules.local.my_unet_model',
        'MyUNetModel')
    assert pipeline.config.scheduler == ('diffusers_modules.local.my_scheduler'
        , 'MyScheduler')
    assert pipeline.__class__.__name__ == 'MyPipeline'
    pipeline = pipeline.to(torch_device)
    images = pipeline('test', num_inference_steps=2, output_type='np')[0]
    assert images.shape == (1, 64, 64, 3)
