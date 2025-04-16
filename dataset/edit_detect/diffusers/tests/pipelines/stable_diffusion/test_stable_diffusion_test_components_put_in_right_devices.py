def test_components_put_in_right_devices(self):
    sd_pipe_with_device_map = StableDiffusionPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5', device_map='balanced',
        torch_dtype=torch.float16)
    assert len(set(sd_pipe_with_device_map.hf_device_map.values())) >= 2
