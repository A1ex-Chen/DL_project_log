def test_integration_logits_no_scale(self):
    path = 'runwayml/stable-diffusion-v1-5'
    lora_id = 'takuma104/lora-test-text-encoder-lora-target'
    pipe = StableDiffusionPipeline.from_pretrained(path, torch_dtype=torch.
        float32)
    pipe.load_lora_weights(lora_id)
    pipe = pipe.to(torch_device)
    self.assertTrue(check_if_lora_correctly_set(pipe.text_encoder),
        'Lora not correctly set in text encoder')
    prompt = 'a red sks dog'
    images = pipe(prompt=prompt, num_inference_steps=30, generator=torch.
        manual_seed(0), output_type='np').images
    expected_slice_scale = np.array([0.074, 0.064, 0.073, 0.0842, 0.069, 
        0.0641, 0.0794, 0.076, 0.084])
    predicted_slice = images[0, -3:, -3:, -1].flatten()
    max_diff = numpy_cosine_similarity_distance(expected_slice_scale,
        predicted_slice)
    assert max_diff < 0.001
    pipe.unload_lora_weights()
    release_memory(pipe)
