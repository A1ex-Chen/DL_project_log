def test_sdxl_1_0_lora_unfusion(self):
    generator = torch.Generator('cpu').manual_seed(0)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        'stabilityai/stable-diffusion-xl-base-1.0')
    lora_model_id = 'hf-internal-testing/sdxl-1.0-lora'
    lora_filename = 'sd_xl_offset_example-lora_1.0.safetensors'
    pipe.load_lora_weights(lora_model_id, weight_name=lora_filename)
    pipe.fuse_lora()
    pipe.enable_model_cpu_offload()
    images = pipe('masterpiece, best quality, mountain', output_type='np',
        generator=generator, num_inference_steps=3).images
    images_with_fusion = images.flatten()
    pipe.unfuse_lora()
    generator = torch.Generator('cpu').manual_seed(0)
    images = pipe('masterpiece, best quality, mountain', output_type='np',
        generator=generator, num_inference_steps=3).images
    images_without_fusion = images.flatten()
    max_diff = numpy_cosine_similarity_distance(images_with_fusion,
        images_without_fusion)
    assert max_diff < 0.0001
    release_memory(pipe)
