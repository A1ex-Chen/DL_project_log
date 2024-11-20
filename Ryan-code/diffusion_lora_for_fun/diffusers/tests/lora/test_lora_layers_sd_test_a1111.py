def test_a1111(self):
    generator = torch.Generator().manual_seed(0)
    pipe = StableDiffusionPipeline.from_pretrained(
        'hf-internal-testing/Counterfeit-V2.5', safety_checker=None).to(
        torch_device)
    lora_model_id = 'hf-internal-testing/civitai-light-shadow-lora'
    lora_filename = 'light_and_shadow.safetensors'
    pipe.load_lora_weights(lora_model_id, weight_name=lora_filename)
    images = pipe('masterpiece, best quality, mountain', output_type='np',
        generator=generator, num_inference_steps=2).images
    images = images[0, -3:, -3:, -1].flatten()
    expected = np.array([0.3636, 0.3708, 0.3694, 0.3679, 0.3829, 0.3677, 
        0.3692, 0.3688, 0.3292])
    max_diff = numpy_cosine_similarity_distance(expected, images)
    assert max_diff < 0.001
    pipe.unload_lora_weights()
    release_memory(pipe)
