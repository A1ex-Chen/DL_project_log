def test_amused_256_fp16(self):
    pipe = AmusedImg2ImgPipeline.from_pretrained('amused/amused-256',
        torch_dtype=torch.float16, variant='fp16')
    pipe.to(torch_device)
    image = load_image(
        'https://huggingface.co/datasets/diffusers/docs-images/resolve/main/open_muse/mountains.jpg'
        ).resize((256, 256)).convert('RGB')
    image = pipe('winter mountains', image, generator=torch.Generator().
        manual_seed(0), num_inference_steps=2, output_type='np').images
    image_slice = image[0, -3:, -3:, -1].flatten()
    assert image.shape == (1, 256, 256, 3)
    expected_slice = np.array([0.998, 0.998, 0.994, 0.9944, 0.996, 0.9908, 
        1.0, 1.0, 0.9986])
    assert np.abs(image_slice - expected_slice).max() < 0.01
