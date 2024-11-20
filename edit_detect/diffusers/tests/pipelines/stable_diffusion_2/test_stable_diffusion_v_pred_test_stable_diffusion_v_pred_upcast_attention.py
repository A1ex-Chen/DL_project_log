def test_stable_diffusion_v_pred_upcast_attention(self):
    sd_pipe = StableDiffusionPipeline.from_pretrained(
        'stabilityai/stable-diffusion-2-1', torch_dtype=torch.float16)
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
    expected_slice = np.array([0.4209, 0.4087, 0.4097, 0.4209, 0.386, 
        0.4329, 0.428, 0.4324, 0.4187])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.05
