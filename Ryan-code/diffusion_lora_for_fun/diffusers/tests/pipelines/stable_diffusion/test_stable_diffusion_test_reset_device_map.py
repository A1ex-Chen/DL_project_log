def test_reset_device_map(self):
    sd_pipe_with_device_map = StableDiffusionPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5', device_map='balanced',
        torch_dtype=torch.float16)
    sd_pipe_with_device_map.reset_device_map()
    assert sd_pipe_with_device_map.hf_device_map is None
    for name, component in sd_pipe_with_device_map.components.items():
        if isinstance(component, torch.nn.Module):
            assert component.device.type == 'cpu'
