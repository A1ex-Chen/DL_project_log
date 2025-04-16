def test_sdxl_0_9_lora_one(self):
    generator = torch.Generator().manual_seed(0)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        'stabilityai/stable-diffusion-xl-base-0.9')
    lora_model_id = 'hf-internal-testing/sdxl-0.9-daiton-lora'
    lora_filename = 'daiton-xl-lora-test.safetensors'
    pipe.load_lora_weights(lora_model_id, weight_name=lora_filename)
    pipe.enable_model_cpu_offload()
    images = pipe('masterpiece, best quality, mountain', output_type='np',
        generator=generator, num_inference_steps=2).images
    images = images[0, -3:, -3:, -1].flatten()
    expected = np.array([0.3838, 0.3482, 0.3588, 0.3162, 0.319, 0.3369, 
        0.338, 0.3366, 0.3213])
    max_diff = numpy_cosine_similarity_distance(expected, images)
    assert max_diff < 0.001
    pipe.unload_lora_weights()
    release_memory(pipe)
