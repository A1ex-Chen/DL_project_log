def test_inpaint_pndm(self):
    sd_pipe = StableDiffusionInpaintPipeline.from_pretrained(
        'runwayml/stable-diffusion-inpainting')
    sd_pipe.scheduler = PNDMScheduler.from_config(sd_pipe.scheduler.config)
    sd_pipe.to(torch_device)
    sd_pipe.set_progress_bar_config(disable=None)
    inputs = self.get_inputs(torch_device)
    image = sd_pipe(**inputs).images[0]
    expected_image = load_numpy(
        'https://huggingface.co/datasets/diffusers/test-arrays/resolve/main/stable_diffusion_inpaint/stable_diffusion_inpaint_pndm.npy'
        )
    max_diff = np.abs(expected_image - image).max()
    assert max_diff < 0.001
