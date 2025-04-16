def test_sdxl_0_9_lora_two(self):
    generator = torch.Generator().manual_seed(0)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        'stabilityai/stable-diffusion-xl-base-0.9')
    lora_model_id = 'hf-internal-testing/sdxl-0.9-costumes-lora'
    lora_filename = 'saijo.safetensors'
    pipe.load_lora_weights(lora_model_id, weight_name=lora_filename)
    pipe.enable_model_cpu_offload()
    images = pipe('masterpiece, best quality, mountain', output_type='np',
        generator=generator, num_inference_steps=2).images
    images = images[0, -3:, -3:, -1].flatten()
    expected = np.array([0.3137, 0.3269, 0.3355, 0.255, 0.2577, 0.2563, 
        0.2679, 0.2758, 0.2626])
    max_diff = numpy_cosine_similarity_distance(expected, images)
    assert max_diff < 0.001
    pipe.unload_lora_weights()
    release_memory(pipe)
