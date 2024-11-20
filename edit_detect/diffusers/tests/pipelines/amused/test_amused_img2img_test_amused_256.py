def test_amused_256(self):
    pipe = AmusedImg2ImgPipeline.from_pretrained('amused/amused-256')
    pipe.to(torch_device)
    image = load_image(
        'https://huggingface.co/datasets/diffusers/docs-images/resolve/main/open_muse/mountains.jpg'
        ).resize((256, 256)).convert('RGB')
    image = pipe('winter mountains', image, generator=torch.Generator().
        manual_seed(0), num_inference_steps=2, output_type='np').images
    image_slice = image[0, -3:, -3:, -1].flatten()
    assert image.shape == (1, 256, 256, 3)
    expected_slice = np.array([0.9993, 1.0, 0.9996, 1.0, 0.9995, 0.9925, 
        0.999, 0.9954, 1.0])
    assert np.abs(image_slice - expected_slice).max() < 0.01
