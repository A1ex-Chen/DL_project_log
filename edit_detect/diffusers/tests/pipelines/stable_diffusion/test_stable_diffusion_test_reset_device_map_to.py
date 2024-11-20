def test_reset_device_map_to(self):
    sd_pipe_with_device_map = StableDiffusionPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5', device_map='balanced',
        torch_dtype=torch.float16)
    sd_pipe_with_device_map.reset_device_map()
    assert sd_pipe_with_device_map.hf_device_map is None
    pipe = sd_pipe_with_device_map.to('cuda')
    _ = pipe('hello', num_inference_steps=2)
