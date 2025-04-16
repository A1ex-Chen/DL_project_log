def test_stable_diffusion_fp16_vs_autocast(self):
    pipe = StableDiffusionPipeline.from_pretrained(
        'CompVis/stable-diffusion-v1-4', torch_dtype=torch.float16)
    pipe = pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    inputs = self.get_inputs(torch_device, dtype=torch.float16)
    image_fp16 = pipe(**inputs).images
    with torch.autocast(torch_device):
        inputs = self.get_inputs(torch_device)
        image_autocast = pipe(**inputs).images
    diff = np.abs(image_fp16.flatten() - image_autocast.flatten())
    assert diff.mean() < 0.02
