def test_pixart_1024(self):
    generator = torch.Generator('cpu').manual_seed(0)
    pipe = PixArtAlphaPipeline.from_pretrained(self.ckpt_id_1024,
        torch_dtype=torch.float16)
    pipe.enable_model_cpu_offload()
    prompt = self.prompt
    image = pipe(prompt, generator=generator, num_inference_steps=2,
        output_type='np').images
    image_slice = image[0, -3:, -3:, -1]
    expected_slice = np.array([0.0742, 0.0835, 0.2114, 0.0295, 0.0784, 
        0.2361, 0.1738, 0.2251, 0.3589])
    max_diff = numpy_cosine_similarity_distance(image_slice.flatten(),
        expected_slice)
    self.assertLessEqual(max_diff, 0.0001)
