def test_max_memory(self):
    no_device_map_image = self.get_pipeline_output_without_device_map()
    sd_pipe_with_device_map = StableDiffusionPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5', device_map='balanced', max_memory
        ={(0): '1GB', (1): '1GB'}, torch_dtype=torch.float16)
    sd_pipe_with_device_map.set_progress_bar_config(disable=True)
    inputs = self.get_inputs()
    device_map_image = sd_pipe_with_device_map(**inputs).images
    max_diff = np.abs(device_map_image - no_device_map_image).max()
    assert max_diff < 0.001
