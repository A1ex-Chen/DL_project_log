def test_stable_diffusion_inpaint_pipeline_fp16(self):
    init_image = load_image(
        'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd2-inpaint/init_image.png'
        )
    mask_image = load_image(
        'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd2-inpaint/mask.png'
        )
    expected_image = load_numpy(
        'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd2-inpaint/yellow_cat_sitting_on_a_park_bench_fp16.npy'
        )
    model_id = 'stabilityai/stable-diffusion-2-inpainting'
    pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id,
        torch_dtype=torch.float16, safety_checker=None)
    pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    pipe.enable_attention_slicing()
    prompt = 'Face of a yellow cat, high resolution, sitting on a park bench'
    generator = torch.manual_seed(0)
    output = pipe(prompt=prompt, image=init_image, mask_image=mask_image,
        generator=generator, output_type='np')
    image = output.images[0]
    assert image.shape == (512, 512, 3)
    assert np.abs(expected_image - image).max() < 0.5
