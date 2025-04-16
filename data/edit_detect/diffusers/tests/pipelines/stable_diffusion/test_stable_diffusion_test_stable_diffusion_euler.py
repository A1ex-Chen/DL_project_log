def test_stable_diffusion_euler(self):
    sd_pipe = StableDiffusionPipeline.from_pretrained(
        'CompVis/stable-diffusion-v1-4').to(torch_device)
    sd_pipe.scheduler = EulerDiscreteScheduler.from_config(sd_pipe.
        scheduler.config)
    sd_pipe.set_progress_bar_config(disable=None)
    inputs = self.get_inputs(torch_device)
    image = sd_pipe(**inputs).images[0]
    expected_image = load_numpy(
        'https://huggingface.co/datasets/diffusers/test-arrays/resolve/main/stable_diffusion_text2img/stable_diffusion_1_4_euler.npy'
        )
    max_diff = np.abs(expected_image - image).max()
    assert max_diff < 0.001
