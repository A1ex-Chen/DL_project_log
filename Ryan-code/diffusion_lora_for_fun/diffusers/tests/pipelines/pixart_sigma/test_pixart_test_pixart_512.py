def test_pixart_512(self):
    generator = torch.Generator('cpu').manual_seed(0)
    pipe = PixArtSigmaPipeline.from_pretrained(self.ckpt_id_512,
        torch_dtype=torch.float16)
    pipe.enable_model_cpu_offload()
    prompt = self.prompt
    image = pipe(prompt, generator=generator, num_inference_steps=2,
        output_type='np').images
    image_slice = image[0, -3:, -3:, -1]
    expected_slice = np.array([0.3477, 0.3882, 0.4541, 0.3413, 0.3821, 
        0.4463, 0.4001, 0.4409, 0.4958])
    max_diff = numpy_cosine_similarity_distance(image_slice.flatten(),
        expected_slice)
    self.assertLessEqual(max_diff, 0.0001)
