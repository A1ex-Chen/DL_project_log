def test_img2img_safety_checker_works(self):
    sd_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5')
    sd_pipe.to(torch_device)
    sd_pipe.set_progress_bar_config(disable=None)
    inputs = self.get_inputs(torch_device)
    inputs['num_inference_steps'] = 20
    inputs['prompt'] = 'naked, sex, porn'
    out = sd_pipe(**inputs)
    assert out.nsfw_content_detected[0
        ], f"Safety checker should work for prompt: {inputs['prompt']}"
    assert np.abs(out.images[0]).sum() < 1e-05
