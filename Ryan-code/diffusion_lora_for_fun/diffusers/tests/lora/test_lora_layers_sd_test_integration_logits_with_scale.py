def test_integration_logits_with_scale(self):
    path = 'runwayml/stable-diffusion-v1-5'
    lora_id = 'takuma104/lora-test-text-encoder-lora-target'
    pipe = StableDiffusionPipeline.from_pretrained(path, torch_dtype=torch.
        float32)
    pipe.load_lora_weights(lora_id)
    pipe = pipe.to(torch_device)
    self.assertTrue(check_if_lora_correctly_set(pipe.text_encoder),
        'Lora not correctly set in text encoder')
    prompt = 'a red sks dog'
    images = pipe(prompt=prompt, num_inference_steps=15,
        cross_attention_kwargs={'scale': 0.5}, generator=torch.manual_seed(
        0), output_type='np').images
    expected_slice_scale = np.array([0.307, 0.283, 0.31, 0.31, 0.3, 0.314, 
        0.336, 0.314, 0.321])
    predicted_slice = images[0, -3:, -3:, -1].flatten()
    max_diff = numpy_cosine_similarity_distance(expected_slice_scale,
        predicted_slice)
    assert max_diff < 0.001
    pipe.unload_lora_weights()
    release_memory(pipe)
