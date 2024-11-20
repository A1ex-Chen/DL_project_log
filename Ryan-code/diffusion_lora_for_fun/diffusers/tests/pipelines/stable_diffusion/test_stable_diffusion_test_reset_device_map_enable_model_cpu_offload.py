def test_reset_device_map_enable_model_cpu_offload(self):
    sd_pipe_with_device_map = StableDiffusionPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5', device_map='balanced',
        torch_dtype=torch.float16)
    sd_pipe_with_device_map.reset_device_map()
    assert sd_pipe_with_device_map.hf_device_map is None
    sd_pipe_with_device_map.enable_model_cpu_offload()
    _ = sd_pipe_with_device_map('hello', num_inference_steps=2)
