def test_amused_512_fp16(self):
    pipe = AmusedImg2ImgPipeline.from_pretrained('amused/amused-512',
        variant='fp16', torch_dtype=torch.float16)
    pipe.to(torch_device)
    image = load_image(
        'https://huggingface.co/datasets/diffusers/docs-images/resolve/main/open_muse/mountains.jpg'
        ).resize((512, 512)).convert('RGB')
    image = pipe('winter mountains', image, generator=torch.Generator().
        manual_seed(0), num_inference_steps=2, output_type='np').images
    image_slice = image[0, -3:, -3:, -1].flatten()
    assert image.shape == (1, 512, 512, 3)
    expected_slice = np.array([0.1536, 0.1767, 0.0227, 0.1079, 0.24, 0.1427,
        0.1511, 0.1564, 0.1542])
    assert np.abs(image_slice - expected_slice).max() < 0.1
