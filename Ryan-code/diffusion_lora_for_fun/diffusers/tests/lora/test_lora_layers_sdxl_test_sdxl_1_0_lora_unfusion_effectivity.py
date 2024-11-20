def test_sdxl_1_0_lora_unfusion_effectivity(self):
    pipe = StableDiffusionXLPipeline.from_pretrained(
        'stabilityai/stable-diffusion-xl-base-1.0')
    pipe.enable_model_cpu_offload()
    generator = torch.Generator().manual_seed(0)
    images = pipe('masterpiece, best quality, mountain', output_type='np',
        generator=generator, num_inference_steps=2).images
    original_image_slice = images[0, -3:, -3:, -1].flatten()
    lora_model_id = 'hf-internal-testing/sdxl-1.0-lora'
    lora_filename = 'sd_xl_offset_example-lora_1.0.safetensors'
    pipe.load_lora_weights(lora_model_id, weight_name=lora_filename)
    pipe.fuse_lora()
    generator = torch.Generator().manual_seed(0)
    _ = pipe('masterpiece, best quality, mountain', output_type='np',
        generator=generator, num_inference_steps=2).images
    pipe.unfuse_lora()
    pipe.unload_lora_weights()
    generator = torch.Generator().manual_seed(0)
    images = pipe('masterpiece, best quality, mountain', output_type='np',
        generator=generator, num_inference_steps=2).images
    images_without_fusion_slice = images[0, -3:, -3:, -1].flatten()
    max_diff = numpy_cosine_similarity_distance(images_without_fusion_slice,
        original_image_slice)
    assert max_diff < 0.001
    release_memory(pipe)
