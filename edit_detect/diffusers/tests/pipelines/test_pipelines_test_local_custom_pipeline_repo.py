def test_local_custom_pipeline_repo(self):
    local_custom_pipeline_path = get_tests_dir('fixtures/custom_pipeline')
    pipeline = DiffusionPipeline.from_pretrained('google/ddpm-cifar10-32',
        custom_pipeline=local_custom_pipeline_path)
    pipeline = pipeline.to(torch_device)
    images, output_str = pipeline(num_inference_steps=2, output_type='np')
    assert pipeline.__class__.__name__ == 'CustomLocalPipeline'
    assert images[0].shape == (1, 32, 32, 3)
    assert output_str == 'This is a local test'
