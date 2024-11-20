def test_img2img_2nd_order(self):
    sd_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5')
    sd_pipe.scheduler = HeunDiscreteScheduler.from_config(sd_pipe.scheduler
        .config)
    sd_pipe.to(torch_device)
    sd_pipe.set_progress_bar_config(disable=None)
    inputs = self.get_inputs(torch_device)
    inputs['num_inference_steps'] = 10
    inputs['strength'] = 0.75
    image = sd_pipe(**inputs).images[0]
    expected_image = load_numpy(
        'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/img2img/img2img_heun.npy'
        )
    max_diff = np.abs(expected_image - image).max()
    assert max_diff < 0.05
    inputs = self.get_inputs(torch_device)
    inputs['num_inference_steps'] = 11
    inputs['strength'] = 0.75
    image_other = sd_pipe(**inputs).images[0]
    mean_diff = np.abs(image - image_other).mean()
    assert mean_diff < 0.05
