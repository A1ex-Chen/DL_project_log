def get_pipeline_output_without_device_map(self):
    sd_pipe = StableDiffusionPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5', torch_dtype=torch.float16).to(
        torch_device)
    sd_pipe.set_progress_bar_config(disable=True)
    inputs = self.get_inputs()
    no_device_map_image = sd_pipe(**inputs).images
    del sd_pipe
    return no_device_map_image
