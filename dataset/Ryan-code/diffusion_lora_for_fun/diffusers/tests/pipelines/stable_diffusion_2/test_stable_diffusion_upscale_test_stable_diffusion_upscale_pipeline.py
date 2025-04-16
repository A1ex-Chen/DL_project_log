def test_stable_diffusion_upscale_pipeline(self):
    image = load_image(
        'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd2-upscale/low_res_cat.png'
        )
    expected_image = load_numpy(
        'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd2-upscale/upsampled_cat.npy'
        )
    model_id = 'stabilityai/stable-diffusion-x4-upscaler'
    pipe = StableDiffusionUpscalePipeline.from_pretrained(model_id)
    pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    pipe.enable_attention_slicing()
    prompt = 'a cat sitting on a park bench'
    generator = torch.manual_seed(0)
    output = pipe(prompt=prompt, image=image, generator=generator,
        output_type='np')
    image = output.images[0]
    assert image.shape == (512, 512, 3)
    assert np.abs(expected_image - image).max() < 0.001
