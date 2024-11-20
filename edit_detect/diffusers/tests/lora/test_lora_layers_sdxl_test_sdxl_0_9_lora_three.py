def test_sdxl_0_9_lora_three(self):
    generator = torch.Generator().manual_seed(0)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        'stabilityai/stable-diffusion-xl-base-0.9')
    lora_model_id = 'hf-internal-testing/sdxl-0.9-kamepan-lora'
    lora_filename = 'kame_sdxl_v2-000020-16rank.safetensors'
    pipe.load_lora_weights(lora_model_id, weight_name=lora_filename)
    pipe.enable_model_cpu_offload()
    images = pipe('masterpiece, best quality, mountain', output_type='np',
        generator=generator, num_inference_steps=2).images
    images = images[0, -3:, -3:, -1].flatten()
    expected = np.array([0.4015, 0.3761, 0.3616, 0.3745, 0.3462, 0.3337, 
        0.3564, 0.3649, 0.3468])
    max_diff = numpy_cosine_similarity_distance(expected, images)
    assert max_diff < 0.005
    pipe.unload_lora_weights()
    release_memory(pipe)
