def test_sdxl_1_0_last_ben(self):
    generator = torch.Generator().manual_seed(0)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        'stabilityai/stable-diffusion-xl-base-1.0')
    pipe.enable_model_cpu_offload()
    lora_model_id = 'TheLastBen/Papercut_SDXL'
    lora_filename = 'papercut.safetensors'
    pipe.load_lora_weights(lora_model_id, weight_name=lora_filename)
    images = pipe('papercut.safetensors', output_type='np', generator=
        generator, num_inference_steps=2).images
    images = images[0, -3:, -3:, -1].flatten()
    expected = np.array([0.5244, 0.4347, 0.4312, 0.4246, 0.4398, 0.4409, 
        0.4884, 0.4938, 0.4094])
    max_diff = numpy_cosine_similarity_distance(expected, images)
    assert max_diff < 0.001
    pipe.unload_lora_weights()
    release_memory(pipe)
