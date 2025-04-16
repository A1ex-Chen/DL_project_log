def test_stable_diffusion_v_pred_euler(self):
    scheduler = EulerDiscreteScheduler.from_pretrained(
        'stabilityai/stable-diffusion-2', subfolder='scheduler')
    sd_pipe = StableDiffusionPipeline.from_pretrained(
        'stabilityai/stable-diffusion-2', scheduler=scheduler)
    sd_pipe = sd_pipe.to(torch_device)
    sd_pipe.enable_attention_slicing()
    sd_pipe.set_progress_bar_config(disable=None)
    prompt = 'A painting of a squirrel eating a burger'
    generator = torch.manual_seed(0)
    output = sd_pipe([prompt], generator=generator, num_inference_steps=5,
        output_type='np')
    image = output.images
    image_slice = image[0, 253:256, 253:256, -1]
    assert image.shape == (1, 768, 768, 3)
    expected_slice = np.array([0.1781, 0.1695, 0.1661, 0.1705, 0.1588, 
        0.1699, 0.2005, 0.1589, 0.1677])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01
