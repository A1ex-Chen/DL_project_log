def test_sdxl_1_0_lora(self):
    generator = torch.Generator('cpu').manual_seed(0)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        'stabilityai/stable-diffusion-xl-base-1.0')
    pipe.enable_model_cpu_offload()
    lora_model_id = 'hf-internal-testing/sdxl-1.0-lora'
    lora_filename = 'sd_xl_offset_example-lora_1.0.safetensors'
    pipe.load_lora_weights(lora_model_id, weight_name=lora_filename)
    images = pipe('masterpiece, best quality, mountain', output_type='np',
        generator=generator, num_inference_steps=2).images
    images = images[0, -3:, -3:, -1].flatten()
    expected = np.array([0.4468, 0.4061, 0.4134, 0.3637, 0.3202, 0.365, 
        0.3786, 0.3725, 0.3535])
    max_diff = numpy_cosine_similarity_distance(expected, images)
    assert max_diff < 0.0001
    pipe.unload_lora_weights()
    release_memory(pipe)
