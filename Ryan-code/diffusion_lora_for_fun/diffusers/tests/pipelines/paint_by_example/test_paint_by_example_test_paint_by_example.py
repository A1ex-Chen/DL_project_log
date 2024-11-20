def test_paint_by_example(self):
    init_image = load_image(
        'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/paint_by_example/dog_in_bucket.png'
        )
    mask_image = load_image(
        'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/paint_by_example/mask.png'
        )
    example_image = load_image(
        'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/paint_by_example/panda.jpg'
        )
    pipe = PaintByExamplePipeline.from_pretrained(
        'Fantasy-Studio/Paint-by-Example')
    pipe = pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    generator = torch.manual_seed(321)
    output = pipe(image=init_image, mask_image=mask_image, example_image=
        example_image, generator=generator, guidance_scale=5.0,
        num_inference_steps=50, output_type='np')
    image = output.images
    image_slice = image[0, -3:, -3:, -1]
    assert image.shape == (1, 512, 512, 3)
    expected_slice = np.array([0.4834, 0.4811, 0.4874, 0.5122, 0.5081, 
        0.5144, 0.5291, 0.529, 0.5374])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01
