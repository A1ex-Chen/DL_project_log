def test_amused_256_fp16(self):
    pipe = AmusedInpaintPipeline.from_pretrained('amused/amused-256',
        variant='fp16', torch_dtype=torch.float16)
    pipe.to(torch_device)
    image = load_image(
        'https://huggingface.co/datasets/diffusers/docs-images/resolve/main/open_muse/mountains_1.jpg'
        ).resize((256, 256)).convert('RGB')
    mask_image = load_image(
        'https://huggingface.co/datasets/diffusers/docs-images/resolve/main/open_muse/mountains_1_mask.png'
        ).resize((256, 256)).convert('L')
    image = pipe('winter mountains', image, mask_image, generator=torch.
        Generator().manual_seed(0), num_inference_steps=2, output_type='np'
        ).images
    image_slice = image[0, -3:, -3:, -1].flatten()
    assert image.shape == (1, 256, 256, 3)
    expected_slice = np.array([0.0735, 0.0749, 0.065, 0.0739, 0.0805, 
        0.0667, 0.0802, 0.0923, 0.0622])
    assert np.abs(image_slice - expected_slice).max() < 0.1
