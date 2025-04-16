def test_load_custom_github(self):
    pipeline = DiffusionPipeline.from_pretrained('google/ddpm-cifar10-32',
        custom_pipeline='one_step_unet', custom_revision='main')
    with torch.no_grad():
        output = pipeline()
    assert output.numel() == output.sum()
    del sys.modules['diffusers_modules.git.one_step_unet']
    pipeline = DiffusionPipeline.from_pretrained('google/ddpm-cifar10-32',
        custom_pipeline='one_step_unet', custom_revision='0.10.2')
    with torch.no_grad():
        output = pipeline()
    assert output.numel() != output.sum()
    assert pipeline.__class__.__name__ == 'UnetSchedulerOneForwardPipeline'
