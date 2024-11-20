def test_amused_512(self):
    pipe = AmusedImg2ImgPipeline.from_pretrained('amused/amused-512')
    pipe.to(torch_device)
    image = load_image(
        'https://huggingface.co/datasets/diffusers/docs-images/resolve/main/open_muse/mountains.jpg'
        ).resize((512, 512)).convert('RGB')
    image = pipe('winter mountains', image, generator=torch.Generator().
        manual_seed(0), num_inference_steps=2, output_type='np').images
    image_slice = image[0, -3:, -3:, -1].flatten()
    assert image.shape == (1, 512, 512, 3)
    expected_slice = np.array([0.1344, 0.0985, 0.0, 0.1194, 0.1809, 0.0765,
        0.0854, 0.1371, 0.0933])
    assert np.abs(image_slice - expected_slice).max() < 0.1
