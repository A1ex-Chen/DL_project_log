def test_amused_512_fp16(self):
    pipe = AmusedInpaintPipeline.from_pretrained('amused/amused-512',
        variant='fp16', torch_dtype=torch.float16)
    pipe.to(torch_device)
    image = load_image(
        'https://huggingface.co/datasets/diffusers/docs-images/resolve/main/open_muse/mountains_1.jpg'
        ).resize((512, 512)).convert('RGB')
    mask_image = load_image(
        'https://huggingface.co/datasets/diffusers/docs-images/resolve/main/open_muse/mountains_1_mask.png'
        ).resize((512, 512)).convert('L')
    image = pipe('winter mountains', image, mask_image, generator=torch.
        Generator().manual_seed(0), num_inference_steps=2, output_type='np'
        ).images
    image_slice = image[0, -3:, -3:, -1].flatten()
    assert image.shape == (1, 512, 512, 3)
    expected_slice = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0025, 0.0])
    assert np.abs(image_slice - expected_slice).max() < 0.003
