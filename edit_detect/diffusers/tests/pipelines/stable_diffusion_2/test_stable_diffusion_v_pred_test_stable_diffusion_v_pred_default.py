def test_stable_diffusion_v_pred_default(self):
    sd_pipe = StableDiffusionPipeline.from_pretrained(
        'stabilityai/stable-diffusion-2')
    sd_pipe = sd_pipe.to(torch_device)
    sd_pipe.enable_attention_slicing()
    sd_pipe.set_progress_bar_config(disable=None)
    prompt = 'A painting of a squirrel eating a burger'
    generator = torch.manual_seed(0)
    output = sd_pipe([prompt], generator=generator, guidance_scale=7.5,
        num_inference_steps=20, output_type='np')
    image = output.images
    image_slice = image[0, 253:256, 253:256, -1]
    assert image.shape == (1, 768, 768, 3)
    expected_slice = np.array([0.1868, 0.1922, 0.1527, 0.1921, 0.1908, 
        0.1624, 0.1779, 0.1652, 0.1734])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01
