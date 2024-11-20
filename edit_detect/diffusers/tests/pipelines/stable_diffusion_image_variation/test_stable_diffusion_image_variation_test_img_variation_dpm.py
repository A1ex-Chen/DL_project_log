def test_img_variation_dpm(self):
    sd_pipe = StableDiffusionImageVariationPipeline.from_pretrained(
        'fusing/sd-image-variations-diffusers')
    sd_pipe.scheduler = DPMSolverMultistepScheduler.from_config(sd_pipe.
        scheduler.config)
    sd_pipe.to(torch_device)
    sd_pipe.set_progress_bar_config(disable=None)
    inputs = self.get_inputs(torch_device)
    inputs['num_inference_steps'] = 25
    image = sd_pipe(**inputs).images[0]
    expected_image = load_numpy(
        'https://huggingface.co/datasets/diffusers/test-arrays/resolve/main/stable_diffusion_imgvar/lambdalabs_variations_dpm_multi.npy'
        )
    max_diff = np.abs(expected_image - image).max()
    assert max_diff < 0.001
