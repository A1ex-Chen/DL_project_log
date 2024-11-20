def test_remote_auto_custom_pipe(self):
    with self.assertRaises(ValueError):
        pipeline = DiffusionPipeline.from_pretrained(
            'hf-internal-testing/tiny-sdxl-custom-all')
    pipeline = DiffusionPipeline.from_pretrained(
        'hf-internal-testing/tiny-sdxl-custom-all', trust_remote_code=True)
    assert pipeline.config.unet == ('diffusers_modules.local.my_unet_model',
        'MyUNetModel')
    assert pipeline.config.scheduler == ('diffusers_modules.local.my_scheduler'
        , 'MyScheduler')
    assert pipeline.__class__.__name__ == 'MyPipeline'
    pipeline = pipeline.to(torch_device)
    images = pipeline('test', num_inference_steps=2, output_type='np')[0]
    assert images.shape == (1, 64, 64, 3)
