def test_lycoris(self):
    generator = torch.Generator().manual_seed(0)
    pipe = StableDiffusionPipeline.from_pretrained('hf-internal-testing/Amixx',
        safety_checker=None, use_safetensors=True, variant='fp16').to(
        torch_device)
    lora_model_id = 'hf-internal-testing/edgLycorisMugler-light'
    lora_filename = 'edgLycorisMugler-light.safetensors'
    pipe.load_lora_weights(lora_model_id, weight_name=lora_filename)
    images = pipe('masterpiece, best quality, mountain', output_type='np',
        generator=generator, num_inference_steps=2).images
    images = images[0, -3:, -3:, -1].flatten()
    expected = np.array([0.6463, 0.658, 0.599, 0.6542, 0.6512, 0.6213, 
        0.658, 0.6485, 0.6017])
    max_diff = numpy_cosine_similarity_distance(expected, images)
    assert max_diff < 0.001
    pipe.unload_lora_weights()
    release_memory(pipe)
