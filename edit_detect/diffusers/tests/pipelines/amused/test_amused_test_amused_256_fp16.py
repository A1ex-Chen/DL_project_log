def test_amused_256_fp16(self):
    pipe = AmusedPipeline.from_pretrained('amused/amused-256', variant=
        'fp16', torch_dtype=torch.float16)
    pipe.to(torch_device)
    image = pipe('dog', generator=torch.Generator().manual_seed(0),
        num_inference_steps=2, output_type='np').images
    image_slice = image[0, -3:, -3:, -1].flatten()
    assert image.shape == (1, 256, 256, 3)
    expected_slice = np.array([0.0554, 0.05129, 0.0344, 0.0452, 0.0476, 
        0.0271, 0.0495, 0.0527, 0.0158])
    assert np.abs(image_slice - expected_slice).max() < 0.007
