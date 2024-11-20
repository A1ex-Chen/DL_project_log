def test_amused_256(self):
    pipe = AmusedPipeline.from_pretrained('amused/amused-256')
    pipe.to(torch_device)
    image = pipe('dog', generator=torch.Generator().manual_seed(0),
        num_inference_steps=2, output_type='np').images
    image_slice = image[0, -3:, -3:, -1].flatten()
    assert image.shape == (1, 256, 256, 3)
    expected_slice = np.array([0.4011, 0.3992, 0.379, 0.3856, 0.3772, 
        0.3711, 0.3919, 0.385, 0.3625])
    assert np.abs(image_slice - expected_slice).max() < 0.003
