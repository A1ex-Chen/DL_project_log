def test_stable_diffusion_pipeline_with_sequential_cpu_offloading_v_pred(self):
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    pipeline_id = 'stabilityai/stable-diffusion-2'
    prompt = 'Andromeda galaxy in a bottle'
    pipeline = StableDiffusionPipeline.from_pretrained(pipeline_id,
        torch_dtype=torch.float16)
    pipeline = pipeline.to(torch_device)
    pipeline.enable_attention_slicing(1)
    pipeline.enable_sequential_cpu_offload()
    generator = torch.manual_seed(0)
    _ = pipeline(prompt, generator=generator, num_inference_steps=5)
    mem_bytes = torch.cuda.max_memory_allocated()
    assert mem_bytes < 2.8 * 10 ** 9
