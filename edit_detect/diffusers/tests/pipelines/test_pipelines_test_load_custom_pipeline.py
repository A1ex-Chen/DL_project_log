def test_load_custom_pipeline(self):
    pipeline = DiffusionPipeline.from_pretrained('google/ddpm-cifar10-32',
        custom_pipeline='hf-internal-testing/diffusers-dummy-pipeline')
    pipeline = pipeline.to(torch_device)
    assert pipeline.__class__.__name__ == 'CustomPipeline'
