def test_amused_512(self):
    pipe = AmusedPipeline.from_pretrained('amused/amused-512')
    pipe.to(torch_device)
    image = pipe('dog', generator=torch.Generator().manual_seed(0),
        num_inference_steps=2, output_type='np').images
    image_slice = image[0, -3:, -3:, -1].flatten()
    assert image.shape == (1, 512, 512, 3)
    expected_slice = np.array([0.996, 0.996, 0.9946, 0.998, 0.9947, 0.9932,
        0.996, 0.9961, 0.9947])
    assert np.abs(image_slice - expected_slice).max() < 0.003
