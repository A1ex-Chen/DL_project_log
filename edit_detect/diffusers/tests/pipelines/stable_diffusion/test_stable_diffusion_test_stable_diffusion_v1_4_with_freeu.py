def test_stable_diffusion_v1_4_with_freeu(self):
    sd_pipe = StableDiffusionPipeline.from_pretrained(
        'CompVis/stable-diffusion-v1-4').to(torch_device)
    sd_pipe.set_progress_bar_config(disable=None)
    inputs = self.get_inputs(torch_device)
    inputs['num_inference_steps'] = 25
    sd_pipe.enable_freeu(s1=0.9, s2=0.2, b1=1.2, b2=1.4)
    image = sd_pipe(**inputs).images
    image = image[0, -3:, -3:, -1].flatten()
    expected_image = [0.0721, 0.0588, 0.0268, 0.0384, 0.0636, 0.0, 0.0429, 
        0.0344, 0.0309]
    max_diff = np.abs(expected_image - image).max()
    assert max_diff < 0.001
