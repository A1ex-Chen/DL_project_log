def test_kohya_sd_v15_with_higher_dimensions(self):
    generator = torch.Generator().manual_seed(0)
    pipe = StableDiffusionPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5', safety_checker=None).to(torch_device)
    lora_model_id = 'hf-internal-testing/urushisato-lora'
    lora_filename = 'urushisato_v15.safetensors'
    pipe.load_lora_weights(lora_model_id, weight_name=lora_filename)
    images = pipe('masterpiece, best quality, mountain', output_type='np',
        generator=generator, num_inference_steps=2).images
    images = images[0, -3:, -3:, -1].flatten()
    expected = np.array([0.7165, 0.6616, 0.5833, 0.7504, 0.6718, 0.587, 
        0.6871, 0.6361, 0.5694])
    max_diff = numpy_cosine_similarity_distance(expected, images)
    assert max_diff < 0.001
    pipe.unload_lora_weights()
    release_memory(pipe)
