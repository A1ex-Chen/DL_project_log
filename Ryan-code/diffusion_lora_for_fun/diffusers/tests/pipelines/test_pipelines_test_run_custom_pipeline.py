def test_run_custom_pipeline(self):
    pipeline = DiffusionPipeline.from_pretrained('google/ddpm-cifar10-32',
        custom_pipeline='hf-internal-testing/diffusers-dummy-pipeline')
    pipeline = pipeline.to(torch_device)
    images, output_str = pipeline(num_inference_steps=2, output_type='np')
    assert images[0].shape == (1, 32, 32, 3)
    assert output_str == 'This is a test'
