def test_amused_512_fp16(self):
    pipe = AmusedPipeline.from_pretrained('amused/amused-512', variant=
        'fp16', torch_dtype=torch.float16)
    pipe.to(torch_device)
    image = pipe('dog', generator=torch.Generator().manual_seed(0),
        num_inference_steps=2, output_type='np').images
    image_slice = image[0, -3:, -3:, -1].flatten()
    assert image.shape == (1, 512, 512, 3)
    expected_slice = np.array([0.9983, 1.0, 1.0, 1.0, 1.0, 0.9989, 0.9994, 
        0.9976, 0.9977])
    assert np.abs(image_slice - expected_slice).max() < 0.003
