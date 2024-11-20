def test_unclip_image_variation_karlo(self):
    input_image = load_image(
        'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/unclip/cat.png'
        )
    expected_image = load_numpy(
        'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/unclip/karlo_v1_alpha_cat_variation_fp16.npy'
        )
    pipeline = UnCLIPImageVariationPipeline.from_pretrained(
        'kakaobrain/karlo-v1-alpha-image-variations', torch_dtype=torch.float16
        )
    pipeline = pipeline.to(torch_device)
    pipeline.set_progress_bar_config(disable=None)
    generator = torch.Generator(device='cpu').manual_seed(0)
    output = pipeline(input_image, generator=generator, output_type='np')
    image = output.images[0]
    assert image.shape == (256, 256, 3)
    assert_mean_pixel_difference(image, expected_image, 15)
