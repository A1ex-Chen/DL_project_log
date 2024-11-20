def test_sdxl_1_0_lora_fusion_efficiency(self):
    generator = torch.Generator().manual_seed(0)
    lora_model_id = 'hf-internal-testing/sdxl-1.0-lora'
    lora_filename = 'sd_xl_offset_example-lora_1.0.safetensors'
    pipe = StableDiffusionXLPipeline.from_pretrained(
        'stabilityai/stable-diffusion-xl-base-1.0', torch_dtype=torch.float16)
    pipe.load_lora_weights(lora_model_id, weight_name=lora_filename,
        torch_dtype=torch.float16)
    pipe.enable_model_cpu_offload()
    start_time = time.time()
    for _ in range(3):
        pipe('masterpiece, best quality, mountain', output_type='np',
            generator=generator, num_inference_steps=2).images
    end_time = time.time()
    elapsed_time_non_fusion = end_time - start_time
    del pipe
    pipe = StableDiffusionXLPipeline.from_pretrained(
        'stabilityai/stable-diffusion-xl-base-1.0', torch_dtype=torch.float16)
    pipe.load_lora_weights(lora_model_id, weight_name=lora_filename,
        torch_dtype=torch.float16)
    pipe.fuse_lora()
    pipe.unload_lora_weights()
    pipe.enable_model_cpu_offload()
    generator = torch.Generator().manual_seed(0)
    start_time = time.time()
    for _ in range(3):
        pipe('masterpiece, best quality, mountain', output_type='np',
            generator=generator, num_inference_steps=2).images
    end_time = time.time()
    elapsed_time_fusion = end_time - start_time
    self.assertTrue(elapsed_time_fusion < elapsed_time_non_fusion)
    release_memory(pipe)
