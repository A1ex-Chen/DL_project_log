def test_sd_load_civitai_empty_network_alpha(self):
    """
        This test simply checks that loading a LoRA with an empty network alpha works fine
        See: https://github.com/huggingface/diffusers/issues/5606
        """
    pipeline = StableDiffusionPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5').to(torch_device)
    pipeline.enable_sequential_cpu_offload()
    civitai_path = hf_hub_download('ybelkada/test-ahi-civitai',
        'ahi_lora_weights.safetensors')
    pipeline.load_lora_weights(civitai_path, adapter_name='ahri')
    images = pipeline('ahri, masterpiece, league of legends', output_type=
        'np', generator=torch.manual_seed(156), num_inference_steps=5).images
    images = images[0, -3:, -3:, -1].flatten()
    expected = np.array([0.0, 0.0, 0.0, 0.002557, 0.020954, 0.001792, 
        0.006581, 0.00591, 0.002995])
    max_diff = numpy_cosine_similarity_distance(expected, images)
    assert max_diff < 0.001
    pipeline.unload_lora_weights()
    release_memory(pipeline)
