def test_stable_diffusion_dpm(self):
    sd_pipe = StableDiffusionPipeline.from_pretrained(
        'stabilityai/stable-diffusion-2-1-base').to(torch_device)
    sd_pipe.scheduler = DPMSolverMultistepScheduler.from_config(sd_pipe.
        scheduler.config, final_sigmas_type='sigma_min')
    sd_pipe.set_progress_bar_config(disable=None)
    inputs = self.get_inputs(torch_device)
    inputs['num_inference_steps'] = 25
    image = sd_pipe(**inputs).images[0]
    expected_image = load_numpy(
        'https://huggingface.co/datasets/diffusers/test-arrays/resolve/main/stable_diffusion_2_text2img/stable_diffusion_2_1_base_dpm_multi.npy'
        )
    max_diff = np.abs(expected_image - image).max()
    assert max_diff < 0.001
